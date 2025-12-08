from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.api as sm
import copy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mvn, gamma as xgamma, norm,multivariate_normal
from scipy.special import gamma, factorial
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import traceback
 
 
 
# Configuración para permitir el uso de toda la GPU disponible
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Hacer visibles todas las GPUs disponibles
    tf.config.set_visible_devices(physical_devices, 'GPU')

    # Configurar para permitir el crecimiento dinámico de la memoria de cada GPU
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No se detectaron GPUs.")

 
 
def simular_logAR1(n, phi, sigma):
   observaciones = np.zeros(n)
   ce = -(sigma**2)/(2*(1-phi**2))
   observaciones[0] =np.exp(np.random.normal(ce, sigma/np.sqrt((1-phi**2))))
   for t in range(1, n):
       #print(observaciones[t-1])
       observaciones[t] = np.exp( (1-phi)*ce + phi * np.log(observaciones[t-1]) + np.random.normal(0, sigma)   )
       #observaciones[t] = (1-phi) +phi * observaciones[t-1] + np.random.normal(0, sigma)
   return observaciones


def simular_X1(n, beta1):
 X1 = (np.random.exponential(scale=1,size=n)**beta1)/gamma(1+beta1)
 return X1

np.random.seed(1)
nsites=25
n_epochs = 25
n_iterations_per_epoch = 500
n_batch_size = 128


 
 
datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste.sort_values('date')
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']

 
loc = pd.read_csv('covariables_guanacaste.csv')
locaciones_completas=loc[(loc['lon']>-85.7) & (loc['lon']<-85.2)  & (loc['lat']>10.3) & (loc['lat']<10.8)]
Z1 = locaciones_completas['lon']  # primera covariable espacial
Z2 = locaciones_completas['lat']   # segunda covariable espacial
Z12 = locaciones_completas['lon']**2  # primera covariable espacial
Z22 = locaciones_completas['lat']**2   # segunda covariable espacial
Z3 = locaciones_completas['alt'] 
Z32 = locaciones_completas['alt']**2 


scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))
scaler = MinMaxScaler()
Z12 = scaler.fit_transform(Z12.values.reshape(-1,1))
scaler = MinMaxScaler()
Z22 = scaler.fit_transform(Z22.values.reshape(-1,1))
scaler = MinMaxScaler()
Z3 = scaler.fit_transform(Z3.values.reshape(-1,1))
scaler = MinMaxScaler()
Z32 = scaler.fit_transform(Z32.values.reshape(-1,1))




# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(locaciones_completas[['lon','lat']]))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_guanacaste)/len(loc))

print(str(nsites) + ' locaciones')
 
def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3


def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 2, ncov)
 return simul_gamma
 
def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)


def quitar_covariable(fila_locacion,gamma_cov):
    suma = -gamma_cov[0]*fila_locacion[0]    
    return np.exp(suma)

cov = cov_original
print('Inicia funciones de previa')
def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar

    return(previas)

parametros = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]
prior = Prior(prior_fun=model_prior, param_names=parametros)
prior_means, prior_stds = prior.estimate_means_and_stds()

def configure_input(forward_dict):
    """Configures dictionary of prior draws and simulated data into BayesFlow format."""

    out_dict = {}

    # standardization sim_data
    sim_data = forward_dict["sim_data"].astype(np.float32)
    #norm_data = (sim_data - sim_mean) / sim_std

    # standardization priors
    params = forward_dict["prior_draws"].astype(np.float32)
    #norm_params = (params - prior_means) / prior_stds

    # remove nan, inf and -inf
    keep_idx = np.all(np.isfinite(sim_data), axis=(1, 2,3,4))
    if not np.all(keep_idx):
        print("Invalid value encountered...removing from batch")

    # add to dict
    out_dict["summary_conditions"] = sim_data[keep_idx]
    out_dict["parameters"] = params[keep_idx]

    return out_dict

def proceso_D4(params, m):
    y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=7.5,size=1)[0] #OJO
    y_train_rho_auxiliar =    np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites+4, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar[nsites]=y_train_phi_auxiliar
    X_train_auxiliar[nsites+1]=y_train_sigma_auxiliar
    X_train_auxiliar[nsites+2]=y_train_beta3_auxiliar
    X_train_auxiliar[nsites+3]=y_train_rho_auxiliar    
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites+4, m).transpose(0, 2, 1)
    return(np.array([X_train_auxiliar]))
time_points=m


simulator_D4 = Simulator(simulator_fun=partial(proceso_D4, m=time_points))
model_D4 = GenerativeModel(prior, simulator_D4, name="simulador_proceso")


data_D4 = model_D4(batch_size=2)
sim_mean_D4 = np.mean(data_D4["sim_data"])
sim_std_D4 = np.std(data_D4["sim_data"])


print('Inicia simulacion')
simul_previa_D4 = model_D4(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')



class CustomLSTM_D4(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = time_points
        features = nsites+4
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps, features)),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
 #               tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )
 
    def call(self, x, **kwargs):
        x = tf.reshape(x, (-1, time_points, nsites+4))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out

def funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,hidden_size, summary_dim,nombre_modelo,n_row):


    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    summary_net_D4 = CustomLSTM_D4(hidden_size, summary_dim)
    inference_net_D4 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS)
    amortizer_D4 = AmortizedPosterior(inference_net_D4, summary_net_D4, name=nombre_modelo)
    trainer_D4 = Trainer(amortizer=amortizer_D4, generative_model=model_D4, memory=False, checkpoint_path = nombre_modelo)
     

    
    
    try:
        history_D4 = trainer_D4.train_offline(simulations_dict=simul_previa_D4,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
        valid_sim_data_raw_D4 = model_D4(batch_size=512)
        valid_sim_data_D4 = trainer_D4.configurator(valid_sim_data_raw_D4)
        posterior_samples_D4 = amortizer_D4.sample(valid_sim_data_D4, n_samples=100)
        fig = diag.plot_recovery(posterior_samples_D4, valid_sim_data_D4["parameters"], param_names=parametros,n_row=n_row)
        fig.savefig(nombre_modelo + '.PNG')
        print('######################################################################')
        print('Finaliza '+ nombre_modelo)
    except:
        print('Falla el '+ nombre_modelo)
        print(traceback.format_exc())

def ajuste_df_covariable(simul,cov,covariable_quitar):
    for sm in range(len(simul['sim_data'])):
        valor_previa = simul['prior_draws'][sm,covariable_quitar]
        for sitio in range(nsites):
            simul['sim_data'][sm][0][0][:,sitio]=simul['sim_data'][sm][0][0][:,sitio]*quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    
    simul['prior_draws'] = simul['prior_draws'][:,0:(covariable_quitar)]
    return(simul)



print('Inicia entrenamiento!')


###################################################################
nombre_modelo='covariables_D4_aplicacion_M7'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=3)

###################################################################
covariable_quitar=6
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M6'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=2)

###################################################################
covariable_quitar=5
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M5'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=2)

###################################################################
covariable_quitar=4
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M4'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=2)

###################################################################
covariable_quitar=3
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M3'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=1)


###################################################################
covariable_quitar=2
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M2'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=1)


###################################################################
covariable_quitar=1
simul_previa_D4=ajuste_df_covariable(simul_previa_D4,cov,covariable_quitar)
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_D4_aplicacion_M1'
entrenamiento=funcion_entrenamiento(simul_previa_D4,cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo,n_row=1)












