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
#     X1(s,t) ~ Weibull
#     X3(s,t) = Inversa-Gamma con copula Gaussiana

# Versión paramétros KAPPA, BETAX, RHO en distribución 
region = "Los Santos"
nombre_parametros = ['posterior_delta','posterior_betax','posterior_rho'] # nombre_parametros = [ r"$\kappa$", r"$\beta_x$", r"$\rho$"]
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

configuraciones_modelos_covariables = {
    '4': {'nombre': 'weibull_covariables_modelo_4', 'num_covs': len(columnas_matriz_diseno_n)+1, 'hidden_size': 1024, 'summary_dim': 128},
    '3': {'nombre': 'weibull_covariables_modelo_3', 'num_covs': len(columnas_matriz_diseno_3)+1, 'hidden_size': 1024, 'summary_dim': 128},
    '2': {'nombre': 'weibull_covariables_modelo_2', 'num_covs': len(columnas_matriz_diseno_2)+1, 'hidden_size': 1024, 'summary_dim': 128},
    '1': {'nombre': 'weibull_covariables_modelo_1', 'num_covs': len(columnas_matriz_diseno_1)+1, 'hidden_size': 1024, 'summary_dim': 128}
}

# Selección de variante de columnas para la matriz de diseño
columnas_matriz_diseno = columnas_matriz_diseno_1
configuracion_covariables = configuraciones_modelos_covariables.get('1')

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
train_design = procesamiento_matriz_diseno(train, columnas_matriz_diseno, poly, scaler)
test_design = procesamiento_matriz_diseno(test, columnas_matriz_diseno, poly, scaler)

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

# Según ChatGPT, train es un equivalente a datos_region_filtrados, pero con la diferencia de que train contiene únicamente las ubicaciones seleccionadas para entrenamiento y filtradas por fecha, mientras que datos_region_filtrados contiene todas las ubicaciones de la región sin filtrar por fecha.
datos_region_filtrados = datos_region.merge(ubic_train[['lon', 'lat']], how='inner', on=['lon', 'lat'])

# ============================================================
# 2. VARIABLES GLOBALES DEL SIMULADOR BAYESFLOW
# ============================================================

GAMMA_SD = 3.5

DELTA_MIN = 0.05
DELTA_MAX = 5.00

BETAX_MIN = 2.05
BETAX_MAX = 150.00

RHO_BUTTOM = 0.05
RHO_UPPPER = 2 * np.max(dist_mat)

def previa_covariables(ncov):
    simul_gamma = np.random.normal(0, GAMMA_SD, ncov)
    return simul_gamma

def calcular_alpha(design_mat, gamma_params):
    alpha_long = np.exp(design_mat.to_numpy() @ gamma_params)
    alpha = alpha_long.reshape(nsites_train, m).T
    return alpha

def model_prior():
    delta = np.random.uniform(DELTA_MIN, DELTA_MAX)
    betax = np.random.uniform(BETAX_MIN, BETAX_MAX)
    rho = np.random.uniform(RHO_BUTTOM, RHO_UPPPER)
    previas = [delta, betax, rho]
    previas = np.array(previas)
    return previas

def model_prior_covariables():
    ncov = train_design.shape[1]
    gammas = previa_covariables(ncov)
    previas = gammas
    return(previas)

nombre_parametros_covariables = train_design.columns
prior = Prior(prior_fun=model_prior, param_names=nombre_parametros)
prior_covariables = Prior(prior_fun=model_prior_covariables, param_names=nombre_parametros_covariables)

def proceso(params, m):
    delta_auxiliar, betax_auxiliar, rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites_rectangular, m))

    X1_auxiliar = simular_X1_weibull(m, delta_auxiliar, nsites = 1).reshape(-1) # (m,)
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
        
    X_conv = np.stack([np.array(X_train_auxiliar_para_convolucion)], axis = -1)
    return X_conv

def proceso_covariables(gamma_params, m):
    delta_auxiliar = np.random.uniform(DELTA_MIN, DELTA_MAX)
    betax_auxiliar = np.random.uniform(BETAX_MIN, BETAX_MAX)
    rho_auxiliar = np.random.uniform(RHO_BUTTOM, RHO_UPPPER)

    alpha_full = calcular_alpha(train_design, gamma_params)
    X1_auxiliar = simular_X1_weibull(m, delta_auxiliar, nsites = 1)
    X3_auxiliar_completo = simular_X3(m, betax_auxiliar, rho_auxiliar, nsites_train, dist_mat)

    X_train_auxiliar = alpha_full * X1_auxiliar * X3_auxiliar_completo

    # Agregar parámetros auxiliares
    X_train_auxiliar = np.column_stack([
        X_train_auxiliar,
        np.repeat(delta_auxiliar, m),
        np.repeat(betax_auxiliar, m),
        np.repeat(rho_auxiliar, m)
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(1, m, nsites_train+3)
    return np.array([X_train_auxiliar])

# ----- Gibbs -------------------------------------------------

def datos_para_red1(parametros,nsites,m,simulacion):
    delta_auxiliar,betax_auxiliar,rho_auxiliar = parametros
    X_train_auxiliar = np.zeros((nsites, m))

    for sitio in range(nsites):
        X_train_auxiliar[sitio] = simulacion[0][0][:,sitio]

    # Agregar parámetros auxiliares
    X_train_auxiliar = np.vstack([
        X_train_auxiliar,
        np.repeat(delta_auxiliar, m),
        np.repeat(betax_auxiliar, m),
        np.repeat(rho_auxiliar, m),
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites+3, m).transpose(0,2,1)
    return np.array([X_train_auxiliar])

def datos_para_red2(gamma_params,nsite_completados,m,simulacion,design_mat):
    X_train_auxiliar = np.zeros((nsites_rectangular, m))
    alpha = calcular_alpha(design_mat, gamma_params)

    sitio_ind = 0
    for sitio in range(nsites_rectangular):
        if np.isnan(ubic_rectangular.aux.values[sitio]):
            X_train_auxiliar[sitio] = np.repeat(0,m)

        else:
            # REVISAR CON CARLOS DIMENSIONES DE MATRIZ DISEÑO Y AUXILIAR
            alpha_auxiliar = alpha[:, sitio_ind]
            auxi = simulacion[0][0][:,sitio_ind]
            sitio_ind +=1
            # REVISAR CON CARLOS LOG
            auxi = np.log(auxi/alpha_auxiliar)
            X_train_auxiliar[sitio] = auxi

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites_rectangular, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    
    for tiempo in range(m):
        matriz_t = X_train_auxiliar[0][tiempo,:].reshape(n_lat,n_lon)
        X_train_auxiliar_para_convolucion.append(matriz_t.tolist())
        
    X_conv = np.stack([np.array(X_train_auxiliar_para_convolucion)], axis = -1)
    return X_conv

def gibbs(n_iter,nombres_parametros,n_covariables,parametros_input,nsites,nsites_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer,amortizer):
    print('###################################')
    print('Inicia Gibbs...')
    print('###################################')
    try:
        resultados_cov = []
        delta_pred = []
        betax_pred = []
        rho_pred= []
        inicio = datetime.now()
        for k in range(n_iter):
            input_red_1 = datos_para_red1(parametros_input,nsites,m,simulacion_proceso,cov)
            valid_sim_data_1['sim_data']=input_red_1
            valid_sim_data_1_c = trainer_covariables.configurator(valid_sim_data_1)
            posterior_samples_covariables= amortizer_covariables.sample(valid_sim_data_1_c, n_samples=1)[0]
            input_red_2 = datos_para_red2(posterior_samples_covariables,nsites_completados,m,simulacion_proceso,cov)
            valid_sim_data_raw_DY['sim_data']=np.expand_dims(input_red_2, axis=0)
            valid_sim_data_raw_DY_c = trainer.configurator(valid_sim_data_raw_DY)
            posterior_delta,posterior_betax,posterior_rho= amortizer.sample(valid_sim_data_raw_DY_c, n_samples=1)[0]

            if posterior_delta < DELTA_MIN:
                posterior_delta = DELTA_MIN
            if posterior_betax < BETAX_MIN:
                posterior_betax = BETAX_MIN
            if posterior_rho < RHO_BUTTOM:
                posterior_rho = RHO_BUTTOM
            
            delta_pred += [posterior_delta]
            betax_pred += [posterior_betax]
            rho_pred+= [posterior_rho]

            parametros_input = [posterior_delta,posterior_betax,posterior_rho]
            resultados_cov.append(posterior_samples_covariables)
            if k%1000==0:
                print('############################################')
                print('Iteracion: '+str(k))
        print('Finaliza Gibbs...')
        fin = datetime.now()
        duracion = fin - inicio
        df_simul = pd.DataFrame({nombres_parametros[0]:delta_pred,nombres_parametros[1]:betax_pred, nombres_parametros[2]:rho_pred})
        df_covariables = pd.DataFrame(resultados_cov,columns = ['posterior_g'+str(z+1) for z in range(n_covariables)])
        resultado_final = pd.concat([df_covariables,df_simul],axis=1)
        with open('estimacion_'+f"{nombre_modelo}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write('Modelo:'+'estimacion_'+f"{nombre_modelo}\n")
            f.write(f"Tiempo de ejecución: {duracion}\n\n")
        return(resultado_final)
    
    except Exception as e:
        # Guardar el error en TXT
        with open('estimacion_'+f"{nombre_modelo}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write('Modelo:'+'estimacion_'+f"{nombre_modelo}\n")
            f.write(f"Error: {type(e).__name__}: {e}\n")
            f.write("######################################################################\n\n")

# Matriz de precipitaciones de entrenamiento (formato 1, m_train, nsites_train)
X_train = train.pivot(index='date', columns=['lat', 'lon'], values='chirps')
X_train = X_train.reindex(columns=pd.MultiIndex.from_frame(ubic_train[['lat', 'lon']]))
X_train = X_train.to_numpy().reshape(1, m_train, nsites_train)
X_train_tf = tf.expand_dims(X_train, axis=0)

# ----- Red ---------------------------------------------------

# MODIFICAR CON LOS DATOS NUESTROS PARA TENER DIMENSION reshape(1, nsites, m).transpose(0, 2, 1)
m = m_train
time_points=m

simulator = Simulator(simulator_fun=partial(proceso, m=time_points))
model = GenerativeModel(prior, simulator, name="simulador_proceso")
simulator_covariables = Simulator(simulator_fun=partial(proceso_covariables, m=time_points))
model_covariables = GenerativeModel(prior_covariables, simulator_covariables, name="simulador_proceso")

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

model_name = "Santos_Weibull_X1_X3"
summary_net = CustomLSTM(1024,128)
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
valid_sim_data_raw_parametros = model(batch_size=1)









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

def funcion_prediccion_covariables(num_covs,hidden_size,summary_dim,nombre_modelo_cov):
    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }

    summary_net_covariables = CustomLSTM_covariables(hidden_size, summary_dim)
    inference_net_covariables  = InvertibleNetwork(num_params=num_covs, num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS, coupling_design='spline')
    amortizer_covariables = AmortizedPosterior(inference_net_covariables, summary_net_covariables, name=nombre_modelo_cov)
    trainer_covariables = Trainer(amortizer=amortizer_covariables, generative_model=model_covariables, memory=False, checkpoint_path = nombre_modelo_cov)
    valid_sim_data_raw = model_covariables(batch_size=1)
    return(amortizer_covariables,trainer_covariables,valid_sim_data_raw)

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

n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
n_posteriores= 10000

delta_init = 0.5
betax_init = 5
rho_init = 0.5

parametros_input = [delta_init,betax_init,rho_init]

# PONERLE ATENCIÓN, HACERLO UNO PARA CADA VARIANTE.

######################################################################################################
amortizer_covariables,trainer_covariables,valid_sim_data_1 = funcion_prediccion_covariables(configuracion_covariables['num_covs'],configuracion_covariables['hidden_size'],configuracion_covariables['summary_dim'],configuracion_covariables['nombre'])
trace = gibbs(n_posteriores,nombre_parametros,configuracion_covariables['num_covs'],parametros_input,nsites_train,nsites_rectangular,m,X_train_tf,train_design,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+configuracion_covariables['nombre']+'.csv')
##############################################################




covariable_quitar=6
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M6'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=5
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M5'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################


covariable_quitar=4
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M4'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=3
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M3'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=2
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M2'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')