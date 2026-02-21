
## Corrida de modelos ABC para Los Santos



from functools import partial
import matplotlib.pyplot as plt
import numpy as np
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
datos_santos.loc[:,'lon']=np.round(datos_santos['lon'],3)
datos_santos.loc[:,'lat']=np.round(datos_santos['lat'],3)
datos_santos=datos_santos.sort_values('date')






loc = datos_santos.groupby(['lat','lon'],as_index=False).agg('count').sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
loc_alt = datos_santos.groupby(['lat','lon','elevation'],as_index=False).agg('count').sort_values(['lat','lon'],ascending=[False,True])[['lon','lat','elevation']]
nsites = loc.shape[0]


datos_santos_train = datos_santos[datos_santos.date<'2020-01-01']
datos_santos_test = datos_santos[datos_santos.date>='2020-01-01']



Z1 = loc['lon']  # primera covariable espacial
Z2 = loc['lat']   # segunda covariable espacial
Z12 = loc['lon']**2  # primera covariable espacial
Z22 = loc['lat']**2   # segunda covariable espacial
Z3 = loc_alt['elevation'] 
Z32 = loc_alt['elevation']**2 
#sns.scatterplot(loc,x='lon',y='lat',hue='elevation',size='elevation',sizes=(20,200))


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
dist_mat = squareform(pdist( loc[['lon','lat']]))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_santos_train)/len(loc))

print(str(nsites) + ' locaciones')
print(str(m) + ' dias de datos de precipitacion para entrenamiento')



# Funciones

def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 3, ncov)
 return simul_gamma

def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)


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


def simular_X3(n,rho,beta3,nsites,dist_mat):
  Sigma = np.exp(-dist_mat / rho)
  Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
  X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
  return 1/X3


def simular_bern(n,p):
  X1 = np.random.binomial(1,p,size=n)
  return X1/p



def quitar_covariable(fila_locacion,gamma_cov):
    suma = -gamma_cov[0]*fila_locacion[0]    
    return np.exp(suma)


def ajuste_df_covariable(simul,cov,covariable_quitar,valor_previa):
    for sitio in range(nsites):
        simul[:,sitio]=simul[:,sitio]*quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    return(simul)




## Previas


def model_prior_cov(cov):
  y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
  return(y_train_gamma_auxiliar)

def model_prior_cov_tiempo():
  covariables_tiempo = previa_covariables(2)
  return(covariables_tiempo)


def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]

    y_train_ber_auxiliar =  np.random.uniform(0.25,0.75,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)

    y_train_beta3_auxiliar = np.random.uniform(2,high=7.5,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_auxiliar = [y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_ber_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]

    previas = np.array(y_train_auxiliar)
    return(previas)



indicadoras_tiempo = pd.DataFrame({'e1':[],'e2':[]})
datos_santos_train_loc1= datos_santos_train[(datos_santos_train['lon']==loc.iloc[0]['lon']) & (datos_santos_train['lat']==loc.iloc[0]['lat'])]
months = (datos_santos_train_loc1.date.values.astype('datetime64[M]').astype(int) % 12) + 1

indicadoras_tiempo['e1'] = np.where(np.isin(months, [8, 9, 10, 11, 12]), 1, 0)
indicadoras_tiempo['e2'] = np.where(np.isin(months, [5, 6, 7]), 1, 0)


## Datos de entrenamiento reales
datos_santos_train = datos_santos_train.sort_values(['lat','lon','date'],ascending=[False,True,True])
datos_entrenamiento = np.zeros((m,nsites))


for sitio in range(nsites):
    lat = loc.iloc[sitio]['lat']
    lon = loc.iloc[sitio]['lon']

    datos_entrenamiento[:,sitio] = datos_santos_train[(datos_santos_train['lon']==lon) & (datos_santos_train['lat']==lat)]['chirps'].values
real =  datos_entrenamiento




def corrida_simulacion(real, cov_original, m, nsites, indicadoras_tiempo, dist_mat,seed):
    filas = []
    nsimul = 10000
    np.random.seed(seed)
    for sss in range(nsimul):
        cov = cov_original
        phi, sigma, ber, beta3, rho = model_prior()
        y_train_gamma_auxiliar = model_prior_cov(cov)
        y_covariables_tiempo = model_prior_cov_tiempo()
        covariables_tiempo = np.exp(indicadoras_tiempo * y_covariables_tiempo).product(axis=1).values

        X_train_auxiliar_D3 = np.zeros((m, nsites))
        X_train_auxiliar_D4 = np.zeros((m, nsites))
        X_train_auxiliar_D5 = np.zeros((m, nsites))
        X_train_auxiliar_D6 = np.zeros((m, nsites))
        X_train_auxiliar_D7 = np.zeros((m, nsites))
        X_train_auxiliar_D8 = np.zeros((m, nsites))

        X3_auxiliar_completo = simular_X3(m, beta3, beta3, nsites, dist_mat)
        X2_auxiliar_fijo = simular_logAR1(m, phi, sigma)
        X1_auxiliar_fijo = simular_bern(m, ber)

        for sitio in range(nsites):
            X3_auxiliar = X3_auxiliar_completo[:, sitio] * covariables_tiempo
            X2_auxiliar = simular_logAR1(m, phi, sigma)
            X1_auxiliar = simular_bern(m, ber)
            covariables_auxiliar = calculo_covariable(cov[sitio, :], y_train_gamma_auxiliar)

            X_train_auxiliar_D3[:, sitio] = X2_auxiliar_fijo * X3_auxiliar * covariables_auxiliar
            X_train_auxiliar_D4[:, sitio] = X2_auxiliar * X3_auxiliar * covariables_auxiliar
            X_train_auxiliar_D5[:, sitio] = X2_auxiliar_fijo * X3_auxiliar * X1_auxiliar_fijo * covariables_auxiliar
            X_train_auxiliar_D6[:, sitio] = X2_auxiliar * X3_auxiliar * X1_auxiliar_fijo * covariables_auxiliar
            X_train_auxiliar_D7[:, sitio] = X2_auxiliar_fijo * X3_auxiliar * X1_auxiliar * covariables_auxiliar
            X_train_auxiliar_D8[:, sitio] = X2_auxiliar * X3_auxiliar * X1_auxiliar * covariables_auxiliar

        simulaciones = [X_train_auxiliar_D3, X_train_auxiliar_D4, X_train_auxiliar_D5,
                        X_train_auxiliar_D6, X_train_auxiliar_D7, X_train_auxiliar_D8]
        cov_iter = [6, 5, 4, 3, 2, 1]

        for i in range(len(cov_iter)):
            for s in range(len(simulaciones)):
                error = ((real - simulaciones[s]) ** 2).sum()
                filas.append({
                    'modelo': 'D' + str(s + 3),
                    'covariables': 'M' + str(cov_iter[i] + 1),
                    'covariables_tiempo': y_covariables_tiempo.tolist(),
                    'cov': y_train_gamma_auxiliar.tolist(),
                    'phi': phi,
                    'sigma': sigma,
                    'bernoulli': ber,
                    'beta3': beta3,
                    'rho': rho,
                    'error': error
                })

            covariable_quitar = cov_iter[i]
            for j in range(len(simulaciones)):
                simulaciones[j] = ajuste_df_covariable(simulaciones[j], cov, covariable_quitar,
                                                      y_train_gamma_auxiliar[covariable_quitar])
            cov = cov_original[:, 0:covariable_quitar]

        for s in range(len(simulaciones)):
            error = ((real - simulaciones[s]) ** 2).sum()
            filas.append({
                'modelo': 'D' + str(s + 3),
                'covariables': 'M1',
                'covariables_tiempo': y_covariables_tiempo.tolist(),
                'cov': y_train_gamma_auxiliar.tolist(),
                'phi': phi,
                'sigma': sigma,
                'bernoulli': ber,
                'beta3': beta3,
                'rho': rho,
                'error': error
            })
    return filas

#############

##############
import os
from multiprocessing import Pool

n_cores = 10
seeds = np.random.randint(0, 1000000, size=n_cores)

with Pool(processes=n_cores) as pool:
    resultados = pool.starmap(
        corrida_simulacion,
        [
            (real, cov_original, m, nsites, indicadoras_tiempo, dist_mat, seeds[i])
            for i in range(n_cores)
        ]
    )


# resultados es una lista de listas, podemos unir todo en una sola lista
filas_totales = [fila for sublist in resultados for fila in sublist]
pd.DataFrame(filas_totales).to_parquet('resultados_abc_los_santos3.parquet')



myobj = datetime.now()


# printing the object itself
print("Object:", myobj)
