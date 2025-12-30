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

datos_precipitacion = pd.read_parquet(r'tablas_precipitaciones.parquet')
datos_santos = datos_precipitacion[datos_precipitacion.region=='Los Santos']
datos_santos['lon']=np.round(datos_santos['lon'],3)
datos_santos['lat']=np.round(datos_santos['lat'],3)
datos_santos=datos_santos.sort_values('date')
loc = datos_santos.groupby(['lat','lon'],as_index=False).agg('count').sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
nsites = loc.shape[0]
Z1 = loc['lon']  # primera covariable espacial
Z2 = loc['lat']   # segunda covariable espacial
Z3 = np.random.randn(nsites)  # tercera covariable espacial


scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))
# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(loc))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov = np.column_stack((np.ones(nsites), Z1, Z2))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_santos)/len(loc))
print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')



scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))



# Funciones
def simular_logAR1(n, phi, sigma):
   observaciones = np.zeros(n)
   ce = -(sigma**2)/(2*(1-phi**2))
   observaciones[0] =np.exp(np.random.normal(ce, sigma/np.sqrt((1-phi**2))))
   for t in range(1, n):
       #print(observaciones[t-1])
       observaciones[t] = np.exp( (1-phi)*ce + phi * np.log(observaciones[t-1]) + np.random.normal(0, sigma)   )
       #observaciones[t] = (1-phi) +phi * observaciones[t-1] + np.random.normal(0, sigma)
   return observaciones

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
 

def simular_X3(n,rho,beta3,nsites,dist_mat):
  Sigma = np.exp(-dist_mat / rho)
  Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
  X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
  return 1/X3


def simular_bern(n,p):
  X1 = np.random.binomial(1,p,size=n)
  return X1/p


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
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    
    y_train_ber_auxiliar =  np.random.uniform(0.25,0.75,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(2,high=7.5,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_auxiliar = [y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_ber_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
parametros_D4= [ r"$\phi$", r"$\sigma$",r"$\beta_3$", r"$\rho$"]

def proceso_D4(params, m):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_ber_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X1_auxiliar=simular_bern(m,y_train_ber_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        auxi = X2_auxiliar*X3_auxiliar*X1_auxiliar
        X_train_auxiliar[sitio] = auxi
    return(X_train_auxiliar)

# Simulación de las variables latentes y observadas
params_simular = [0.6,2,0.6,5,0.6]
datos_simulados = proceso_D4(params_simular, m)


# Metodologia: ABC
n_simulaciones = 100
print('Inicia corrida de ' + str(n_simulaciones) + ' simulaciones ABC')

filas = []

for s in range(n_simulaciones):
    phi, sigma, ber, beta3, rho = model_prior_D4()

    a = proceso_D4([phi, sigma, ber, beta3, rho], m)
    error = ((datos_simulados - a) ** 2).sum()

    filas.append({
        'phi': phi,
        'sigma': sigma,
        'bernoulli': ber,
        'beta3': beta3,
        'rho': rho,
        'error': error
    })

simulaciones = pd.DataFrame(filas)
simulaciones.to_csv('simulaciones_ABC_D4_los_santos.csv', index=False)
print('Finaliza corrida de ' + str(n_simulaciones) + ' simulaciones ABC')