from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import copy

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
region = "Los Santos"
ruta_datos = "tablas_precipitaciones.parquet"
fecha_corte = pd.Timestamp("2018-12-31")

# Configuración épocas, entrenamiento, bloques
n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
block_size = 12800
total_sims = n_batch_size*n_iterations_per_epoch

# ----- Matriz de diseño -------------------------------------

columnas_matriz_diseno_n = ['lat', 'lon', 'lat^2', 'lon^2', 'sin_1', 'cos_1', 'sin_2', 'cos_2']
columnas_matriz_diseno_3 = ['lat', 'lon',          'lon^2', 'sin_1', 'cos_1', 'sin_2', 'cos_2']
columnas_matriz_diseno_2 = ['lat', 'lon', 'lat^2',          'sin_1', 'cos_1', 'sin_2', 'cos_2']
columnas_matriz_diseno_1 = ['lat', 'lon',                   'sin_1', 'cos_1', 'sin_2', 'cos_2']

covariables_all = ['const'] + columnas_matriz_diseno_n

# ----- Preparación de datos ---------------------------------

datos_precipitacion = pd.read_parquet(ruta_datos)
datos_region = preparar_datos_region(datos_precipitacion, region)
train, test, nsites_train, nsites_test, nsites_total, ubic_train, ubic_test, ubic, m_train, m_test = dividir_train_test(datos_region, fecha_corte, frac=0.80, seed=1234)
dist_mat = calcular_matriz_distancias(ubic_train)

# Ordenar para alpha_long.reshape(nsites, m).T
train = (
    train
    .sort_values(
        ["lat", "lon", "date"],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
    .copy()
)

test = (
    test
    .sort_values(
        ["lat", "lon", "date"],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
    .copy()
)

poly, scaler = preprocesamiento_escalar_datos(train)
train_design = procesamiento_matriz_diseno(train, columnas_matriz_diseno_n, poly, scaler)
test_design = procesamiento_matriz_diseno(test, columnas_matriz_diseno_n, poly, scaler)

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

GAMMA_SD = 5.0

DELTA_MIN = 0.05
DELTA_MAX = 5.00

BETA3_MIN = 2.05
BETA3_MAX = 150.00

# Evita rho = 0 numéricamente
RHO_BUTTOM = 0.05
RHO_UPPPER = 2 * np.max(dist_mat)

def previa_covariables(ncov):
    simul_gamma = np.random.normal(0, GAMMA_SD, ncov)
    return simul_gamma

def calcular_alpha(design_mat, gamma_params_all, subconjunto_covariables):
    # Determinar indices de covariables del subconjunto elegido
    indices = [covariables_all.index(col) for col in subconjunto_covariables]

    # Matriz de diseño y vector gamma de subconjunto eleligo
    design_aux = design_mat[subconjunto_covariables]
    gamma_aux = np.asarray(gamma_params_all)[indices]

    alpha_long = np.exp(design_aux.to_numpy() @ gamma_aux)
    alpha = alpha_long.reshape(nsites_train, m).T

    return alpha

def model_prior_covariables():
    ncov = train_design.shape[1]
    gammas = previa_covariables(ncov)
    previas = gammas
    return(previas)

nombre_parametros_covariables = train_design.columns
prior_covariables = Prior(prior_fun=model_prior_covariables, param_names=nombre_parametros_covariables)

def proceso_covariables(gamma_params_all, m):
    delta_auxiliar = np.random.uniform(DELTA_MIN, DELTA_MAX)
    betax_auxiliar = np.random.uniform(BETA3_MIN, BETA3_MAX)
    rho_auxiliar = np.random.uniform(RHO_BUTTOM, RHO_UPPPER)

    alpha_full = calcular_alpha(train_design, gamma_params_all, covariables_all)
    X1_auxiliar = simular_X1_lognormal(m, delta_auxiliar, nsites = 1)
    X3_auxiliar_completo = simular_X3(m, betax_auxiliar, rho_auxiliar, nsites_train, dist_mat)

    X_train_auxiliar = alpha_full * X1_auxiliar * X3_auxiliar_completo

    # Agregar parámetros auxiliares
    X_train_auxiliar = np.column_stack([
        X_train_auxiliar,
        np.repeat(delta_auxiliar, m),
        np.repeat(betax_auxiliar, m),
        np.repeat(rho_auxiliar, m)
    ]).astype(np.float32)

    X_train_auxiliar = X_train_auxiliar.reshape(1, m, nsites_train+3)
    return np.array([X_train_auxiliar])

# ----- Red ---------------------------------------------------

m = m_train
time_points=m
simulator_covariables = Simulator(simulator_fun=partial(proceso_covariables, m=time_points))
model_covariables = GenerativeModel(prior_covariables, simulator_covariables, name="simulador_proceso")
data = model_covariables(batch_size=2)

class CustomLSTM_covariables(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = time_points
        features =  nsites_train+3
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps, features)),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                # tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )
 
    def call(self, x, **kwargs):
        x = tf.reshape(x, (-1, time_points,  nsites_train+3))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out


COUPLING_NET_SETTINGS = {
    # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}

def ajustar_covariables(simul, design_mat, subconjunto_covariables):
    simul_aux = copy.deepcopy(simul)
    indices = [covariables_all.index(col) for col in subconjunto_covariables]

    for sm in range(len(simul["sim_data"])):
        gamma_all = simul["prior_draws"][sm, :]

        alpha_full = calcular_alpha(design_mat, gamma_all, covariables_all)
        alpha_sub = calcular_alpha(design_mat, gamma_all, subconjunto_covariables)

        factor = alpha_sub / alpha_full

        simul_aux["sim_data"][sm, 0, 0, :, :nsites_train] *= factor

    simul_aux["prior_draws"] = simul["prior_draws"][:, indices]

    return simul_aux

def crear_configurador(subconjunto_covariables):
    indices = [covariables_all.index(col) for col in subconjunto_covariables]

    def configurator(forward_dict):
        sim_data = copy.deepcopy(forward_dict["sim_data"]).astype(np.float32)
        params_all = forward_dict["prior_draws"].astype(np.float32)

        for sm in range(len(sim_data)):
            gamma_all = params_all[sm]

            alpha_full = calcular_alpha(
                train_design,
                gamma_all,
                covariables_all
            )

            alpha_sub = calcular_alpha(
                train_design,
                gamma_all,
                subconjunto_covariables
            )

            factor = alpha_sub / alpha_full

            sim_data[sm, 0, 0, :, :nsites_train] *= factor

        return {
            "summary_conditions": sim_data,
            "parameters": params_all[:, indices]
        }

    return configurator

def entrenar_multiples_modelos_covariables(sim_block_actual,modelos_config,valid_parametros,ultima,duracion_historica,nombre_modelo):
    
    # 1. Inicializar todos los modelos con sus propios amortizers y trainers
    print("Inicializando redes neuronales...")

    configurador_modelo = crear_configurador(modelos_config['covariables'])
    num_covs = len(modelos_config['covariables'])
    n_row = modelos_config['n_row']
    summary_net_covariables = CustomLSTM_covariables(modelos_config['hidden_size'], modelos_config['summary_dim'])
    inference_net_covariables = InvertibleNetwork(
        num_params=num_covs,
        num_coupling_layers=4,
        coupling_settings=COUPLING_NET_SETTINGS,
        coupling_design='spline'
    )
    
    amortizer_covariables = AmortizedPosterior(
        inference_net_covariables,
        summary_net_covariables,
        name=nombre_modelo
    )
    
    trainer_covariables = Trainer(
    amortizer=amortizer_covariables,
    generative_model=model_covariables,
    configurator=configurador_modelo,
    memory=False,
    checkpoint_path=nombre_modelo
    )
    
    inicio = datetime.now()
    print("######################################################################")
    print("Iniciando entrenamiento...")
    history = trainer_covariables.train_offline(
    simulations_dict=sim_block_actual,
    epochs=n_epochs,
    batch_size=n_batch_size,
    early_stopping=True,
    validation_sims=128
    )
    fin = datetime.now()
    duracion = fin - inicio
    
    if ultima:
        # Generar data de validación nueva
        valid_sim_data_raw = model_covariables(batch_size=128)
        valid_sim_data = trainer_covariables.configurator(valid_sim_data_raw)
        posterior_samples = amortizer_covariables.sample(valid_sim_data, n_samples=100)

        # Generar y guardar gráfico
        fig = diag.plot_recovery(
            posterior_samples,
            valid_sim_data["parameters"],
            param_names=valid_parametros,
            # xlabel="Real",
            # ylabel="Estimado",
            n_row=n_row
        )
        fig.savefig(nombre_modelo + ".PNG")
        duracion = duracion_historica+duracion
        # Guardar en TXT
        with open(f"{nombre_modelo}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write(f"Modelo: {nombre_modelo}\n")
            f.write(f"Tiempo de ejecución: {duracion}\n\n")
            

    return duracion

# --- CONFIGURACIÓN Y EJECUCIÓN ---
print('Inicia preparación de modelos!')

# Definimos el diccionario con la configuración de cada modelo (en orden descendente)
configuracion_modelos = {
    'lognormal_covariables_modelo_4': {
        'covariables': ['const'] + columnas_matriz_diseno_n,
        'hidden_size': 1024,
        'summary_dim': 128,
        'n_row': 3
    },
    'lognormal_covariables_modelo_3': {
        'covariables': ['const'] + columnas_matriz_diseno_3,
        'hidden_size': 1024,
        'summary_dim': 128,
        'n_row': 3
    },
    'lognormal_covariables_modelo_2': {
        'covariables': ['const'] + columnas_matriz_diseno_2,
        'hidden_size': 1024,
        'summary_dim': 128,
        'n_row': 3
    },
    'lognormal_covariables_modelo_1': {
        'covariables': ['const'] + columnas_matriz_diseno_1,
        'hidden_size': 1024,
        'summary_dim': 128,
        'n_row': 3
    }
}

def train_multiple_models(configuraciones):
    from datetime import timedelta
    duraciones = {
        k: timedelta(0)
        for k in configuraciones
    }
    for i in range(0, total_sims, block_size):
        print(f"\n--- Generando Bloque {i//block_size + 1}: {i} → {i+block_size} ---")
        # Número real de simulaciones de este bloque
        n_sims_bloque = min(block_size, total_sims - i)
        # Indica si este es el último bloque
        ultima = (i + n_sims_bloque >= total_sims)

        simul = model_covariables(batch_size=n_sims_bloque)
        for k, config in configuracion_modelos.items():
            covariables_modelo = config['covariables']
            try:
                duraciones[k] = entrenar_multiples_modelos_covariables(
                    simul,
                    config,
                    covariables_modelo,
                    ultima,
                    duraciones.get(k),
                    k
                )
            except Exception as e:
                # Guardar el error en TXT
                with open(f"{k}.txt", "a") as f:
                    f.write("######################################################################\n")
                    f.write(f"Modelo: {k}\n")
                    f.write(f"Error: {type(e).__name__}: {e}\n")
                    f.write("######################################################################\n\n")

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
    train_multiple_models(configuracion_modelos)

if __name__ == "__main__":
    main()