from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer

# ============================================================
# 0. FUNCIONES AUXILIARES
# ============================================================

from funciones import (
    preparar_datos_region,
    dividir_train_test,
    calcular_matriz_distancias,
    preprocesamiento_escalar_datos,
    procesamiento_matriz_diseno,
    simular_X1_lognormal,
    simular_X1_weibull, 
    simular_X3
    )

# ============================================================
# 1. CONFIGURACION
# ============================================================

# Variante:
#     Y = X1(s,t) * X3(s,t)
#     X1(s,t) ~ Lognormal
#     X3(s,t) = Inversa-Gamma con copula Gaussiana

# Versión paramétros KAPPA, BETAX, RHO en distribución 
nombre_modelo = "Santos_Lognormal_X1_X3"
nombre_parametros = [ r"$\delta$", r"$\beta_x$", r"$\rho$"]
region = "Los Santos"
ruta_datos = "tablas_precipitaciones.parquet"
fecha_corte = pd.Timestamp("2018-12-31")

# Configuración épocas, entrenamiento, bloques
n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
n_block_size = 12800

# ----- Preparación de datos ---------------------------------

datos_precipitacion = pd.read_parquet(ruta_datos)
datos_region = preparar_datos_region(datos_precipitacion, region)
train, test, nsites_train, nsites_test, nsites_total, ubic_train, ubic_test, ubic, m_train, m_test = dividir_train_test(datos_region, fecha_corte, frac=0.80, seed=1234)
dist_mat = calcular_matriz_distancias(ubic_train)

# Ordenar para alpha_long.reshape(nsites, m).T
# train = (
#     train
#     .sort_values(
#         ["lat", "lon", "date"],
#         ascending=[False, True, True],
#     )
#     .reset_index(drop=True)
#     .copy()
# )

# test = (
#     test
#     .sort_values(
#         ["lat", "lon", "date"],
#         ascending=[False, True, True],
#     )
#     .reset_index(drop=True)
#     .copy()
# )

# poly, scaler = preprocesamiento_escalar_datos(train)
# train_design = procesamiento_matriz_diseno(train, seleccion_matriz_diseno, poly, scaler)
# test_design = procesamiento_matriz_diseno(test, seleccion_matriz_diseno, poly, scaler)

# Grilla rectangular
ubic_rectangular = (
    pd.MultiIndex.from_product(
        [ubic_train["lat"].unique(), ubic_train["lon"].unique()],
        names=["lat", "lon"]
    )
    .to_frame(index=False)
)

ubic_aux = ubic_train
ubic_aux['aux'] = 1
ubic_rectangular = ubic_rectangular.merge(ubic_aux, on = ['lon','lat'], how = 'left').sort_values(['lat','lon'], ascending=[False,True])

nsites_rectangular = len(ubic_rectangular)
n_lon = ubic_rectangular['lon'].nunique()
n_lat = ubic_rectangular['lat'].nunique()

# ============================================================
# 2. VARIABLES GLOBALES DEL SIMULADOR BAYESFLOW
# ============================================================

# GAMMA_SD = 5.0

DELTA_MIN = 0.05
DELTA_MAX = 5.00

BETAX_MIN = 2.05
BETAX_MAX = 150.00

RHO_BUTTOM = 0.05
RHO_UPPPER = 2 * np.max(dist_mat)

def model_prior():
    delta = np.random.uniform(DELTA_MIN, DELTA_MAX)
    betax = np.random.uniform(BETAX_MIN, BETAX_MAX)
    rho = np.random.uniform(RHO_BUTTOM, RHO_UPPPER)
    previas = [delta, betax, rho]
    previas = np.array(previas)
    return previas

prior = Prior(prior_fun=model_prior, param_names=nombre_parametros)

def proceso(params, m):
    delta_auxiliar, betax_auxiliar, rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites_rectangular, m), dtype=np.float32)
    # float32 para reducir uso de memoria

    X1_auxiliar = simular_X1_lognormal(m, delta_auxiliar, nsites = 1).reshape(-1) # (m,)
    X3_auxiliar_completo = simular_X3(m, betax_auxiliar, rho_auxiliar, nsites_train, dist_mat) # (m,n)

    sitio_ind = 0
    for sitio in range(nsites_rectangular):
        if np.isnan(ubic_rectangular.aux.values[sitio]):
            X_train_auxiliar[sitio] = np.repeat(0,m)

        else:
            X3_auxiliar = X3_auxiliar_completo[:,sitio_ind]
            X_train_auxiliar[sitio] = X1_auxiliar * X3_auxiliar
            sitio_ind +=1

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites_rectangular, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    
    for tiempo in range(m):
        matriz_t = X_train_auxiliar[0][tiempo,:].reshape(n_lat,n_lon)
        X_train_auxiliar_para_convolucion.append(matriz_t.tolist())
        
    X_conv = np.stack([np.array(X_train_auxiliar_para_convolucion)], axis = -1).astype(np.float32)
    return X_conv

# ----- Red ---------------------------------------------------

m = m_train
time_points = m
simulator = Simulator(simulator_fun=partial(proceso, m=time_points))
model = GenerativeModel(prior, simulator, name="simulador_proceso")
data = model(batch_size=1)

from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,n_lat, n_lon, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )

    def call(self, x, **kwargs):
        #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
        out = self.LSTM(x)
        return out

COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}

def train_multiple_models(models,
                          total_sims=n_iterations_per_epoch * n_batch_size,
                          block_size=n_block_size):

    # Initialize each model with its own trainer and amortizer
    trainers = {}
    amortizers = {}
    for model_name, (n1, n2) in models.items():
        # Build summary and inference networks
        summary_net = CustomLSTM(n1, n2)
        inference_net = InvertibleNetwork(
            num_params=3,
            num_coupling_layers=10,
            coupling_settings=COUPLING_NET_SETTINGS,
            coupling_design='spline'
        )
        amortizer = AmortizedPosterior(
            inference_net,
            summary_net,
            name=model_name
        )
        trainer = Trainer(
            amortizer=amortizer,
            generative_model=model,
            memory=False,
            checkpoint_path=model_name
        )
        trainers[model_name] = trainer
        amortizers[model_name] = amortizer

        start_time = datetime.now()
        print("Starting step-by-step training model...")

        # Iterate over simulation blocks
        for i in range(0, total_sims, block_size):
            print(f"Block {i//block_size + 1}: {i} → {i+block_size}")

            # Generate one block of simulations
            sim_block = model(batch_size=block_size)

            # Train each model using the same block
            print(f"Training {model_name} with block {i//block_size + 1}")
            
            history = trainer.train_offline(
                    simulations_dict=sim_block,
                    epochs=n_epochs,
                    batch_size=n_batch_size,
                    early_stopping=True,
                    validation_sims=128
                )
       
        end_time = datetime.now()
        duration = end_time - start_time

        valid_sim_data_raw = model(batch_size=256)
        valid_sim_data = trainers[model_name].configurator(valid_sim_data_raw)
        posterior_samples = amortizer.sample(valid_sim_data, n_samples=100)

        # Save recovery plot
        fig = diag.plot_recovery(
            posterior_samples,
            valid_sim_data["parameters"],
            param_names=nombre_parametros,
            n_col=3
        )
        fig.savefig(model_name + ".PNG")

        # Save results to TXT
        with open(f"{model_name}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Start: {start_time}\n")
            f.write(f"End: {end_time}\n")
            f.write(f"Execution time: {duration}\n\n")


lista_modelos = {f"{nombre_modelo}": (1024,128)}

def revisar_GPU():
    # Configuración para permitir el uso de toda la GPU disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # Hacer visibles todas las GPUs disponibles
        tf.config.set_visible_devices(physical_devices, 'GPU')

        # Configurar para permitir el crecimiento dinámico de la memoria de cada GPU
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('Gpus detectados!')
    else:
        print("No se detectaron GPUs.")

def main():
    revisar_GPU()
    train_multiple_models(lista_modelos)

if __name__ == "__main__":
    main()