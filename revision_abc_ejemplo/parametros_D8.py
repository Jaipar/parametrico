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





n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
 
 
datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']
datos_guanacaste=datos_guanacaste.sort_values('date')
loc = datos_guanacaste.groupby(['lat','lon'],as_index=False).agg(casa = ('id','count')).sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
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


Z1 = locaciones_completas['lon'] 
Z2 = locaciones_completas['lat']   
Z3 = np.random.randn(nsites)  

scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))


# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(locaciones_completas))  
cov = np.column_stack((np.ones(nsites), Z1, Z2))
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
    Indicator = np.zeros((nsite_completados, m))
    
    
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
    X1_auxiliar=simular_X1(m,0.5)
    sitio_ind = 0
    for sitio in range(nsite_completados):
        if np.isnan(loc_all_completar.aux.values[sitio]):
            X_train_auxiliar[sitio] = np.repeat(0,m)
            Indicator[sitio] = np.repeat(0,m)
        else:
            X3_auxiliar=X3_auxiliar_completo[:,sitio_ind]
            X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
            sitio_ind +=1
            auxi = np.log(X2_auxiliar*X3_auxiliar*X1_auxiliar)
            cuantil_75 = np.quantile(auxi,0.75)
            X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
            Indicator[sitio] = np.where(auxi<cuantil_75,1,0)

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    Indicator = Indicator.reshape(1, nsite_completados, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    Indicator_para_convolucion = []
    
    for tiempo in range(m):
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(n_lat,n_lon)
        matriz_ind = Indicator[0][tiempo,:].reshape(n_lat,n_lon)
        X_train_auxiliar_para_convolucion.append(matriz55.tolist())
        Indicator_para_convolucion.append(matriz_ind.tolist())
        
    X_conv = np.stack(
    [

        np.array(X_train_auxiliar_para_convolucion),
        np.array(Indicator_para_convolucion)
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
            [   tf.keras.layers.Input((timesteps,n_lat, n_lon, 2)),
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
 
from datetime import datetime


n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
 


def train_multiple_models(models,
                          total_sims=n_iterations_per_epoch * n_batch_size,
                          block_size=12800):

    # Initialize each model with its own trainer and amortizer
    trainers = {}
    amortizers = {}
    for model_name, (n1, n2) in models.items():
        # Build summary and inference networks
        summary_net = CustomLSTM(n1, n2)
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
        trainers[model_name] = trainer
        amortizers[model_name] = amortizer

    start_time = datetime.now()
    print("Starting step-by-step training with multiple models...")

    # Iterate over simulation blocks
    for i in range(0, total_sims, block_size):
        print(f"Block {i//block_size + 1}: {i} → {i+block_size}")

        # Generate one block of simulations
        sim_block = model(batch_size=block_size)

        # Train each model using the same block
        for model_name, trainer in trainers.items():
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

    # Final validation for each model
    for model_name, amortizer in amortizers.items():
        valid_sim_data_raw = model(batch_size=512)
        valid_sim_data = trainers[model_name].configurator(valid_sim_data_raw)
        posterior_samples = amortizer.sample(valid_sim_data, n_samples=100)

        # Save recovery plot
        fig = diag.plot_recovery(
            posterior_samples,
            valid_sim_data["parameters"],
            param_names=parametros,
            xlabel="True",
            ylabel="Estimated",
            n_col=2
        )
        fig.savefig(model_name + ".PNG")

        # Save results to TXT
        with open(f"{model_name}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Start: {start_time}\n")
            f.write(f"End: {end_time}\n")
            f.write(f"Execution time: {duration}\n\n")

    print("######################################################################")
    print("Finished training all models")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    print(f"Total time: {duration}")


modelos = {
    'parametros_D8_1': (128,128),
    'parametros_D8_2': (128,1024),
    'parametros_D8_3': (1024,128),
    'parametros_D8_4': (1024,1024),
    'parametros_D8_6': (1000,2000)
}

train_multiple_models(modelos)
