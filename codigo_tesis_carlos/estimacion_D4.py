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
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
import requests
#from calculadora_de_error import calculadora_de_error
 
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

def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3


def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 1, ncov)
 return simul_gamma
 
def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)


def quitar_covariable(fila_locacion,gamma_cov):
    suma = -gamma_cov[0]*fila_locacion[0]    
    return np.exp(suma)

# AJUSTAR
def datos_para_red1(parametros,nsites,m,simulacion,cov_matriz):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = parametros

    X_train_auxiliar = np.zeros((nsites, m))
    
    #Indicator  = np.zeros((nsites,m))
    for sitio in range(nsites):
        auxi=simulacion[0][0][:,sitio]
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
                # indicador de valores extremos
       # Indicator[sitio] = np.where(
        #    auxi < cuantil_75,
       #     1,
        #    0)
        

    # agregar parámetros auxiliares
    X_train_auxiliar = np.vstack([
        X_train_auxiliar,
        np.repeat(y_train_phi_auxiliar, m),
        np.repeat(y_train_sigma_auxiliar, m),
        np.repeat(y_train_beta3_auxiliar, m),
        np.repeat(y_train_rho_auxiliar, m)
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(
        1,
        nsites+4,
        m
    ).transpose(0,2,1)

    return np.array([X_train_auxiliar])

def datos_para_red2(covariables,nsite_completados,m,simulacion,cov_matriz):
    X_train_auxiliar = np.zeros((nsite_completados, m)) # 2
    #Indicator = np.zeros((nsite_completados, m))
    sitio_ind = 0
    for sitio in range(nsite_completados):
        if np.isnan(loc_all_completar.aux.values[sitio]):
            X_train_auxiliar[sitio] = np.repeat(0,m)
            #Indicator[sitio] = np.repeat(0,m)
        else:
            covariables_auxiliar = calculo_covariable(cov_matriz[sitio_ind,:],covariables)
            auxi= simulacion[0][0][:,sitio_ind]
            sitio_ind +=1
            auxi = np.log(auxi/covariables_auxiliar)
            cuantil_75 = np.quantile(auxi,0.75)
            X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
            #Indicator[sitio] = np.where(auxi<cuantil_75,1,0)

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    #Indicator = Indicator.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    Indicator_para_convolucion = []
    
    for tiempo in range(m):
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(n_lat,n_lon)
        #matriz_ind = Indicator[0][tiempo,:].reshape(n_lat,n_lon)
        X_train_auxiliar_para_convolucion.append(matriz55.tolist())
        #Indicator_para_convolucion.append(matriz_ind.tolist())
        
    X_conv = np.stack(
    [

        np.array(X_train_auxiliar_para_convolucion),
        #np.array(Indicator_para_convolucion)
    ],
    axis=-1
    )
    return X_conv




def gibbs(n_iter,nombres_parametros,n_covariables,parametros_input,nsites,nsites_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer,amortizer):
    print('###################################')
    print('Inicia Gibbs...')
    print('###################################')
    try:
        resultados_cov = []
        p1_pred = []
        p2_pred = []
        beta3_pred = []
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
            # Cambio
            p1,p2,posterior_beta3,posterior_rho= amortizer.sample(valid_sim_data_raw_DY_c, n_samples=1)[0]
        
            if posterior_beta3<1:
                posterior_beta3 = 1.05
            if posterior_rho<0:
                posterior_rho = 0.05
            if p1<-0.85:
                p1=-0.85
            if p2<0:
                p2=0.05
            if p1>0.85:
                p1=0.85
            
            p1_pred += [p1]
            p2_pred += [p2]
            beta3_pred += [posterior_beta3]
            rho_pred+= [posterior_rho]
            #print([k,p1,p2,posterior_beta3,posterior_rho],posterior_samples_covariables)
            parametros_input = [p1,p2,posterior_beta3,posterior_rho]
            resultados_cov.append(posterior_samples_covariables)
            #print('Iteracion: '+str(k)+' p1: '+str(p1)+' p2: '+str(p2)+' beta3: '+str(posterior_beta3)+' rho: '+str(posterior_rho))
            if k%1000==0:
                print('############################################')
                print('Iteracion: '+str(k))
        print('Finaliza Gibbs...')
        fin = datetime.now()
        duracion = fin - inicio
        df_simul = pd.DataFrame({nombres_parametros[0]:p1_pred,nombres_parametros[1]:p2_pred, nombres_parametros[2]:beta3_pred,nombres_parametros[3]:rho_pred})
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

datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']
datos_guanacaste=datos_guanacaste.sort_values('date')
loc = pd.read_csv('covariables_guanacaste.csv')
loc = loc.sort_values(['lat','lon'],ascending=[False,True])[['lon','lat','alt']]
# 1
test_loc = loc.sample(int(len(loc) * 0.25), random_state=777)
# Without test data
locaciones_completas = loc[
    ~pd.MultiIndex.from_frame(loc[['lat', 'lon']]).isin(
        pd.MultiIndex.from_frame(test_loc[['lat', 'lon']])
    )
]
nsites=len(locaciones_completas)
# 1


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
dist_mat = squareform(pdist(locaciones_completas[['lon','lat']]))
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_guanacaste)/len(loc))

print(str(nsites) + ' locations')
print(str(m) + ' in time')


loc_all = (
    pd.MultiIndex.from_product(
        [loc["lat"].unique(), loc["lon"].unique()],
        names=["lat", "lon"]
    )
    .to_frame(index=False)
)
# Use locaciones_completas to ommit test also
locaciones_completas_aux=locaciones_completas
locaciones_completas_aux['aux'] = 1
loc_all_completar = loc_all.merge(locaciones_completas_aux,on = ['lon','lat'],how = 'left').sort_values(['lat','lon'],ascending=[False,True])
nsite_completados = len(loc_all_completar)

datos_guanacaste_filtrados=datos_guanacaste.merge(locaciones_completas[['lon','lat']],how='inner',on =['lon','lat'])


# Para la estimacion de covariables
X_train_guanacaste= np.zeros((nsites, m))
Indicator  = np.zeros((nsites,m))
for sitio in range(nsites):
    lon_actual=locaciones_completas.iloc[sitio].lon
    lat_actual=locaciones_completas.iloc[sitio].lat
    auxi= datos_guanacaste_filtrados[(datos_guanacaste_filtrados.lon==lon_actual)&(datos_guanacaste_filtrados.lat==lat_actual)].sort_values('date')['chirps'].values
    cuantil_75 = np.quantile(auxi,0.75)
    X_train_guanacaste[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
X_train_guanacaste = X_train_guanacaste.reshape(1, nsites, m).transpose(0, 2, 1)

df_resultados_final = pd.DataFrame({})

#############
# Red de parametros
############


def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    
    y_train_sigma_auxiliar= np.random.uniform(0.05,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_auxiliar = [y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
 
parametros= [ r"$\phi$", r"$\sigma$",r"$\beta_3$", r"$\rho$"]
prior = Prior(prior_fun=model_prior, param_names=parametros)
prior_means, prior_stds = prior.estimate_means_and_stds()


# 2
n_lon=loc_all_completar['lon'].nunique()
n_lat=loc_all_completar['lat'].nunique()

def proceso(params, m):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsite_completados, m)) # 2
    #Indicator = np.zeros((nsite_completados, m))
    
    
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
   
    
    sitio_ind = 0
    for sitio in range(nsite_completados):
        if np.isnan(loc_all_completar.aux.values[sitio]):
            X_train_auxiliar[sitio] = np.repeat(0,m)
            #Indicator[sitio] = np.repeat(0,m)
        else:
            X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
            X3_auxiliar=X3_auxiliar_completo[:,sitio_ind]
            #X1_auxiliar=simular_X1(m,0.5)
            sitio_ind +=1
            auxi = np.log(X2_auxiliar*X3_auxiliar)#*X1_auxiliar)
            cuantil_75 = np.quantile(auxi,0.75)
            X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
            #Indicator[sitio] = np.where(auxi<cuantil_75,1,0)

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    #Indicator = Indicator.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    #Indicator_para_convolucion = []
    
    for tiempo in range(m):
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(n_lat,n_lon)
        #matriz_ind = Indicator[0][tiempo,:].reshape(n_lat,n_lon)
        X_train_auxiliar_para_convolucion.append(matriz55.tolist())
        #Indicator_para_convolucion.append(matriz_ind.tolist())
        
    X_conv = np.stack(
    [

        np.array(X_train_auxiliar_para_convolucion),
        #np.array(Indicator_para_convolucion)
    ],
    axis=-1
    )
    return X_conv
time_points=m
simulator = Simulator(simulator_fun=partial(proceso, m=time_points))
model = GenerativeModel(prior, simulator, name="simulador_proceso")
data = model(batch_size=1)
# 2



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

model_name = 'parametros_D4_3_padding_sim_apli'
summary_net = CustomLSTM(1024,128)
inference_net = InvertibleNetwork(
    num_params=4,
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


#############
# Redes de covariables
############
cov = cov_original
print('Inicia funciones de previa')
def model_prior_covariables():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar

    return(previas)

parametros_covariables = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]
prior_covariables = Prior(prior_fun=model_prior_covariables, param_names=parametros_covariables)
prior_means, prior_stds = prior_covariables.estimate_means_and_stds()

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



def proceso_covariables(params, m):
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    
   
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        #X1_auxiliar=simular_X1(m,0.5)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
                # indicador de valores extremos
        #Indicator[sitio] = np.where(
        #    auxi < cuantil_75,
        #    1,
        #    0)
        
    # concatenar indicadores como nuevas variables
    #X_train_auxiliar = np.concatenate(
    #    (X_train_auxiliar, Indicator),
    #    axis=0
    #)

    # agregar parámetros auxiliares
    X_train_auxiliar = np.vstack([
        X_train_auxiliar,
        np.repeat(y_train_phi_auxiliar, m),
        np.repeat(y_train_sigma_auxiliar, m),
        np.repeat(y_train_beta3_auxiliar, m),
        np.repeat(y_train_rho_auxiliar, m)
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(
        1,
        nsites+4,
        m
    ).transpose(0,2,1)

    return np.array([X_train_auxiliar])
time_points=m


simulator_covariables = Simulator(simulator_fun=partial(proceso_covariables, m=time_points))
model_covariables = GenerativeModel(prior_covariables, simulator_covariables, name="simulador_proceso")


data = model(batch_size=2)
sim_mean = np.mean(data["sim_data"])
sim_std = np.std(data["sim_data"])


class CustomLSTM_covariables(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = time_points
        features =  nsites+4
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
        x = tf.reshape(x, (-1, time_points,  nsites+4))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out


COUPLING_NET_SETTINGS = {
    # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}


def funcion_prediccion_covariables(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,hidden_size, summary_dim,nombre_modelo):


    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    summary_net_covariables = CustomLSTM_covariables(hidden_size, summary_dim)
    inference_net_covariables  = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS,coupling_design='spline')
    amortizer_covariables = AmortizedPosterior(inference_net_covariables, summary_net_covariables, name=nombre_modelo)
    trainer_covariables = Trainer(amortizer=amortizer_covariables, generative_model=model_covariables, memory=False, checkpoint_path = nombre_modelo)
    valid_sim_data_raw = model_covariables(batch_size=1)
    #valid_sim_data_DY = trainer_DY.configurator(valid_sim_data_raw_DY)
    return(amortizer_covariables,trainer_covariables,valid_sim_data_raw)





n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
n_posteriores=10000


phi_init = 0.5
sigma_init = 1
beta3_init = 5
rho_init = 0.5
simulacion_proceso=tf.expand_dims(X_train_guanacaste, axis=0)
parametros_input = [phi_init,sigma_init,beta3_init,rho_init]
nombres_parametros = ['posterior_phi','posterior_sigma','posterior_beta3','posterior_rho']


######################################################################################################
nombre_modelo='covariables_D4_aplicacion_M7'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D4',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)
trace=gibbs(n_posteriores,nombres_parametros,7,parametros_input,nsites,nsite_completados,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_parametros,trainer,amortizer)
trace.to_csv('trace_'+nombre_modelo+'.csv')
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