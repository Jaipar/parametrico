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
# importing datetime module
from datetime import datetime
 
# now is a method in datetime module is
# used to retrieve current data,time
myobj = datetime.now()
 
 
# printing the object itself
print("Object:", myobj)
 
 
 
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
n_iterations_per_epoch = 1000
n_batch_size = 128
 
 
datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste.sort_values('date')
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']
loc = datos_guanacaste.groupby(['lat','lon'],as_index=False).agg(casa = ('id','count')).sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
locaciones_completas=loc[(loc['lon']>-85.7) & (loc['lon']<-85.2)  & (loc['lat']>10.3) & (loc['lat']<10.8)]
Z1 = locaciones_completas['lon']  # primera covariable espacial
Z2 = locaciones_completas['lat']   # segunda covariable espacial
Z3 = np.random.randn(nsites)  # tercera covariable espacial

scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))





# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(locaciones_completas))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov = np.column_stack((np.ones(nsites), Z1, Z2))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_guanacaste)/len(loc))

print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')



 
def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3
def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 3, ncov)
 return simul_gamma
 
def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)
 


 
def model_prior_D4():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.95,size=1)[0]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(2,high=7.5,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_auxiliar = [y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
parametros_D4= [ r"$\phi$", r"$\sigma$",r"$\beta_3$", r"$\rho$"]
prior_D4 = Prior(prior_fun=model_prior_D4, param_names=parametros_D4)
prior_means_D4, prior_stds_D4 = prior_D4.estimate_means_and_stds()
 

##################


def proceso_D4(params, m):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        auxi = np.log(X2_auxiliar*X3_auxiliar)
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m
simulator_D4 = Simulator(simulator_fun=partial(proceso_D4, m=time_points))
model_D4 = GenerativeModel(prior_D4, simulator_D4, name="simulador_proceso")
data_D4 = model_D4(batch_size=1)

#####
nombre_modelo = 'parametros_D4ab'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM_D4(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
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
    
summary_net_D4 = CustomLSTM_D4(128,1024)

 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
 

import pickle
print('Inicia simulacion')
simul_previa_D4 = model_D4(batch_size=n_iterations_per_epoch*n_batch_size)
# try:
#     with open('simulacion_y_dividido_guanacaste_cuadrado_128k.pickle', 'wb') as f:
#       # Pickle the 'data' dictionary using the highest protocol available.   
#         pickle.dump(simul_previa, f, pickle.HIGHEST_PROTOCOL)
       
# except:
#    print('Falla la descarga de los datos')
print('Termina simulacion')
myobj=datetime.now()
print(myobj)

inference_net_D4 = InvertibleNetwork(num_params=len(parametros_D4) ,num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4 = AmortizedPosterior(inference_net_D4, summary_net_D4, name=nombre_modelo)
trainer_D4 = Trainer(amortizer=amortizer_D4, generative_model=model_D4, memory=False, checkpoint_path = nombre_modelo)
history_D4 = trainer_D4.train_offline(simulations_dict=simul_previa_D4,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4 = model_D4(batch_size=512)
valid_sim_data_D4 = trainer_D4.configurator(valid_sim_data_raw_D4)
posterior_samples_D4 = amortizer_D4.sample(valid_sim_data_D4, n_samples=100)
try:
    fig = diag.plot_recovery(posterior_samples_D4, valid_sim_data_D4["parameters"], param_names=parametros_D4, xlabel= "Real",ylabel= "Estimado",n_col=2)
    fig.savefig(nombre_modelo + '.PNG')
except:
    print('Falla')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
myobj= datetime.now()
print(myobj)



#####
nombre_modelo = 'parametros_D4bb'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM_D4(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )

    def call(self, x, **kwargs):
        #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
        out = self.LSTM(x)
        return out
    
summary_net_D4 = CustomLSTM_D4()

 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
 



inference_net_D4 = InvertibleNetwork(num_params=len(parametros_D4) ,num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4 = AmortizedPosterior(inference_net_D4, summary_net_D4, name=nombre_modelo)
trainer_D4 = Trainer(amortizer=amortizer_D4, generative_model=model_D4, memory=False, checkpoint_path = nombre_modelo)
history_D4 = trainer_D4.train_offline(simulations_dict=simul_previa_D4,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4 = model_D4(batch_size=512)
valid_sim_data_D4 = trainer_D4.configurator(valid_sim_data_raw_D4)
posterior_samples_D4 = amortizer_D4.sample(valid_sim_data_D4, n_samples=100)
try:
    fig = diag.plot_recovery(posterior_samples_D4, valid_sim_data_D4["parameters"], param_names=parametros_D4, xlabel= "Real",ylabel= "Estimado",n_col=2)
    fig.savefig(nombre_modelo + '.PNG')
except:
    print('Falla')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
myobj= datetime.now()
print(myobj)



#####
nombre_modelo = 'parametros_D4cb'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM_D4(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
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
    
summary_net_D4 = CustomLSTM_D4()

 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
 

inference_net_D4 = InvertibleNetwork(num_params=len(parametros_D4) ,num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4 = AmortizedPosterior(inference_net_D4, summary_net_D4, name=nombre_modelo)
trainer_D4 = Trainer(amortizer=amortizer_D4, generative_model=model_D4, memory=False, checkpoint_path = nombre_modelo)
history_D4 = trainer_D4.train_offline(simulations_dict=simul_previa_D4,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4 = model_D4(batch_size=512)
valid_sim_data_D4 = trainer_D4.configurator(valid_sim_data_raw_D4)
posterior_samples_D4 = amortizer_D4.sample(valid_sim_data_D4, n_samples=100)
try:
    fig = diag.plot_recovery(posterior_samples_D4, valid_sim_data_D4["parameters"], param_names=parametros_D4, xlabel= "Real",ylabel= "Estimado",n_col=2)
    fig.savefig(nombre_modelo + '.PNG')
except:
    print('Falla')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
myobj= datetime.now()
print(myobj)


#####
nombre_modelo = 'parametros_D4db'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM_D4(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
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
    
summary_net_D4 = CustomLSTM_D4(1024,128)

 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
 

inference_net_D4 = InvertibleNetwork(num_params=len(parametros_D4) ,num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4 = AmortizedPosterior(inference_net_D4, summary_net_D4, name=nombre_modelo)
trainer_D4 = Trainer(amortizer=amortizer_D4, generative_model=model_D4, memory=False, checkpoint_path = nombre_modelo)
history_D4 = trainer_D4.train_offline(simulations_dict=simul_previa_D4,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4 = model_D4(batch_size=512)
valid_sim_data_D4 = trainer_D4.configurator(valid_sim_data_raw_D4)
posterior_samples_D4 = amortizer_D4.sample(valid_sim_data_D4, n_samples=100)
try:
    fig = diag.plot_recovery(posterior_samples_D4, valid_sim_data_D4["parameters"], param_names=parametros_D4, xlabel= "Real",ylabel= "Estimado",n_col=2)
    fig.savefig(nombre_modelo + '.PNG')
except:
    print('Falla')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
myobj= datetime.now()
print(myobj)