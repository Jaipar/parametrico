# Modelo Approximate Bayesian Computation (ABC)
# from functools import partial
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import statsmodels as sm
# import statsmodels.api as sm
# import copy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma, norm, multivariate_normal
from scipy.special import logit, expit
# from scipy.special import gamma, factorial
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
# importing datetime module
from datetime import datetime
import os

# Cargar datos de precipitaciones CHIRPS
datos_precipitacion = pd.read_parquet('tablas_precipitaciones.parquet')

# Filtrar datos para la región de Los Santos y preparar covariables espaciales
datos_santos = datos_precipitacion.loc[datos_precipitacion['region'] == 'Los Santos'].copy()
datos_santos['lon'] = datos_santos['lon'].round(3)
datos_santos['lat'] = datos_santos['lat'].round(3)

# Preprar covariables temporales
datos_santos['date'] = pd.to_datetime(datos_santos['date'])
datos_santos = datos_santos.sort_values('date')

# Dataframe con las ubicaciones únicas
ubic = datos_santos.groupby(['lat','lon','elevation'],as_index=False).agg('count').sort_values(['lat','lon'],ascending=[False,True])[['lon','lat','elevation']]
nsites = ubic.shape[0]

# Conjunto de entrenamiento y prueba
datos_santos_train = datos_santos[datos_santos['date']< pd.to_datetime('2020-01-01')]
datos_santos_test = datos_santos[datos_santos['date'] >= pd.to_datetime('2020-01-01')]

# Propuestas a partir del modelo Yadav et al. (2022)
# Procesos Y_t(s), t = {1, 2, ..., n} sean copias i.i.d del proceso Y(s) tal que:
# Y_t(s) = alpha(s) * X_1t(s) * X_2t(s) * X_3t(s) para procesos no negativos e independientes entre sí.

# X_1t(s) es un proceso de ruido i.i.d con función de distribución F_1 con media 1.
# X_2t(s) es especialmente constante e i.i.d con función de distribución F_2 con media 1.
# Se asume F_1 y F_2 con cola Weibull.
# X_3t(s) es un proceso espacial no trivial con cópula subyacente C_X_3. Distribución marginal F_3 con media 1 y cola regular.
# alpha(s) es una función de covariables espaciales, es decir, alpha(s) = exp(gamma_0 + gamma_1 * Z_1(s) + ... + gamma_p * Z_p(s)) donde Z_i(s) son covariables espaciales y gamma_i son parámetros desconocidos.

# Covariables espaciales
Z1 = ubic['lon'] # primera covariable espacial
Z2 = ubic['lat'] # segunda covariable espacial
Z12 = ubic['lon']**2 # primera covariable espacial al cuadrado
Z22 = ubic['lat']**2 # segunda covariable espacial al cuadrado
Z3 = ubic['elevation'] # tercera covariable espacial
Z32 = ubic['elevation']**2 # tercera covariable espacial al cuadrado
# sns.scatterplot(ubic,x='lon',y='lat',hue='elevation',size='elevation',sizes=(20,200))

## Escalar covariables entre 0 y 1
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

# Matriz de distancia entre ubicaciones
dist_mat = squareform(pdist(ubic[['lon','lat']])) # Dimensiones: nsites x nsites
# np.save('dist_mat.npy', dist_mat)
# Matriz de diseño para covariables espaciales
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))  # Dimensiones: nsites x 7)
# Rango superior para rho (parámetro de la función de correlación espacial), como el doble de la distancia máxima entre ubicaciones
rho_upper_range = 2*np.max(dist_mat)
# Número de días de datos de precipitación para entrenamiento
m = int(len(datos_santos_train)/nsites)

# Funciones

# Previa para gammas (para cálculo de alpha(s)) y otros parámetros sin restricciones sobre la recta real
def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 3, ncov)
 return simul_gamma

# Cálculo de alpha(s) mediante exp( gamma_0 + gamma_1 * Z_1(s) + ... + gamma_p * Z_p(s) )
def calculo_covariable(fila_locacion, gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+= gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)

# Cálculo de proceso con distribución Bernoulli para X_1t(s)
def simular_bern(n,p):
  X1 = np.random.binomial(1,p,size=n)
  return X1/p

def simular_pt(indicadora_mes_transitorio, indicadora_mes_lluvioso, e0, e1, e2):
  pt = expit(e0 + e1*indicadora_mes_transitorio + e2*indicadora_mes_lluvioso)
  return pt

meses_secos = [1, 2, 3, 12]
meses_transitorios = [4, 11]
meses_lluviosos = [5, 6, 7, 8, 9, 10]

def simular_pt(mes_t, p_seco, p_transitorio, p_lluvioso):    
    pt = np.zeros_like(mes_t, dtype=float)
    pt[np.isin(mes_t, meses_secos)] = p_seco
    pt[np.isin(mes_t, meses_transitorios)] = p_transitorio
    pt[np.isin(mes_t, meses_lluviosos)] = p_lluvioso
    return pt

# Cálculo de proceso AR(1) para X_2t(s)
def simular_logAR1(n, phi, sigma):
   observaciones = np.zeros(n)
   ce = -(sigma**2) / ( 2*(1-(phi**2)) )
   observaciones[0] = np.exp( np.random.normal(ce, sigma/np.sqrt((1-(phi**2)))) )
   for t in range(1, n):
       # print(observaciones[t-1])
       observaciones[t] = np.exp( (1-phi)*ce + phi*np.log(observaciones[t-1]) + np.random.normal(0, sigma) )
       # log(observaciones[t]) = (1-phi)*ce + phi*log(observaciones[t-1]) + np.random.normal(0, sigma)
   return observaciones

# Calculo de proceso para X_3t(s) con cópula subyacente C_X_3 y distribución marginal F_3 con media 1 y cola regular (Gamma Inversa)
def simular_X3(n, rho, beta3, nsites, dist_mat):
  Sigma = np.exp(-dist_mat / rho)
  Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
  X3 = (gamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
  return 1/X3

# !!!!!!!!! REVISAR SI REALMENTE SE QUITAN LAS COVARIABLES DE ESTA MANERA, O SI SE DEBERÍA SIMULAR DE NUEVO EL PROCESO CON LA COVARIABLE QUITADA
def quitar_covariable(fila_locacion,gamma_cov):
    suma = -gamma_cov[0]*fila_locacion[0]    
    return np.exp(suma)

# !!!!!!!!! REVISAR SI REALMENTE SE QUITAN LAS COVARIABLES DE ESTA MANERA, O SI SE DEBERÍA SIMULAR DE NUEVO EL PROCESO CON LA COVARIABLE QUITADA
def ajuste_df_covariable(simul,cov,covariable_quitar,valor_previa):
    for sitio in range(nsites):
        simul[:,sitio] = simul[:,sitio]*quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    return(simul)

# Retirar aporte de la covariable por eliminarse de la simulación
def ajuste_df_covariable(simul, cov, covariable_quitar, valor_previa):
    for sitio in range(nsites):
        simul[:, sitio] = simul[:,sitio] * np.exp(-valor_previa * cov[sitio, covariable_quitar])
    return simul

## Previas
def model_prior_cov(cov):
  y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
  return(y_train_gamma_auxiliar)

def model_prior_cov_tiempo():
  covariables_tiempo = previa_covariables(2)
  return(covariables_tiempo)

def model_prior():
    # Phi: Grado de autocorrelación temporal (AR(1))
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85)
    # Parámetro Bernoulli para X_1t(s) para meses secos
    y_train_ber_seco =  np.random.uniform(0,0.40)
    # Parámetro Bernoulli para X_1t(s) para meses transitorios
    y_train_ber_transitorio =  np.random.uniform(0.20,0.60)
    # Parámetro Bernoulli para X_1t(s) para meses lluviosos
    y_train_ber_lluvioso =  np.random.uniform(0.60,1)
    # Sigma: Desviación estándar del proceso logAR(1) para X_2t(s)
    y_train_sigma_auxiliar = np.random.uniform(0,3)
    # Beta3: Parámetro de la distribución marginal de X_3t(s) Gamma Inversa
    y_train_beta3_auxiliar = np.random.uniform(2,7.5)
    # Parámetro de la correlación espacial (rho) para matriz de distancia
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range)

    y_train_auxiliar = [y_train_phi_auxiliar,
                        y_train_sigma_auxiliar,
                        y_train_ber_seco,
                        y_train_ber_transitorio,
                        y_train_ber_lluvioso,
                        y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
    
    previas = np.array(y_train_auxiliar)
    return(previas)

# Indicadoras asociadas a estaciones climáticas
meses = datos_santos_train[(datos_santos_train['lon']==ubic.iloc[0]['lon']) & (datos_santos_train['lat']==ubic.iloc[0]['lat'])]['date'].dt.month

# indicadoras_tiempo = pd.DataFrame({'e1':[],'e2':[]})

# datos_santos_train_loc1= datos_santos_train[(datos_santos_train['lon']==ubic.iloc[0]['lon']) & (datos_santos_train['lat']==ubic.iloc[0]['lat'])]
# months = (datos_santos_train_loc1.date.values.astype('datetime64[M]').astype(int) % 12) + 1

# indicadoras_tiempo['e1'] = np.where(np.isin(months, [8, 9, 10, 11, 12]), 1, 0)
# indicadoras_tiempo['e2'] = np.where(np.isin(months, [5, 6, 7]), 1, 0)

## Datos de entrenamiento reales
datos_santos_train = datos_santos_train.sort_values(['lat','lon','date'],ascending=[False,True,True])
datos_entrenamiento = np.zeros((m,nsites))

for sitio in range(nsites):
    lat = ubic.iloc[sitio]['lat']
    lon = ubic.iloc[sitio]['lon']

    datos_entrenamiento[:,sitio] = datos_santos_train[(datos_santos_train['lon']==lon) & (datos_santos_train['lat']==lat)]['chirps'].values
real =  datos_entrenamiento


def corrida_simulacion(real, cov_original, m, nsites, meses, dist_mat,seed):
    filas = []
    print(f"PID {os.getpid()} seed {seed}")
    print("Cantidad de simulaciones: ", nsimul)
    np.random.seed(seed)
    for sss in range(nsimul):
        cov = cov_original.copy()
        phi, sigma, ber_seco, ber_transitorio, ber_lluvioso, beta3, rho = model_prior()
        y_train_gamma_auxiliar = model_prior_cov(cov)
        # y_covariables_tiempo = model_prior_cov_tiempo()
        # covariables_tiempo = np.exp(indicadoras_tiempo * y_covariables_tiempo).product(axis=1).values

        X_train_auxiliar_D3 = np.zeros((m, nsites))
        X_train_auxiliar_D4 = np.zeros((m, nsites))
        X_train_auxiliar_D5 = np.zeros((m, nsites))
        X_train_auxiliar_D6 = np.zeros((m, nsites))
        X_train_auxiliar_D7 = np.zeros((m, nsites))
        X_train_auxiliar_D8 = np.zeros((m, nsites))

        X3_auxiliar_completo = simular_X3(m, rho, beta3, nsites, dist_mat)
        X2_auxiliar_fijo = simular_logAR1(m, phi, sigma)
        p_t = simular_pt(meses, ber_seco, ber_transitorio, ber_lluvioso)
        X1_auxiliar_fijo = simular_bern(m, p_t)

        for sitio in range(nsites):
            X3_auxiliar = X3_auxiliar_completo[:, sitio] #* covariables_tiempo
            X2_auxiliar = simular_logAR1(m, phi, sigma)
            X1_auxiliar = simular_bern(m, p_t)
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
                    # 'covariables_tiempo': y_covariables_tiempo.tolist(),
                    'cov': y_train_gamma_auxiliar.tolist(),
                    'phi': phi,
                    'sigma': sigma,
                    # 'bernoulli': ber,
                    'p_seco': ber_seco,
                    'p_transitorio': ber_transitorio,
                    'p_lluvioso': ber_lluvioso,
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
                # 'covariables_tiempo': y_covariables_tiempo.tolist(),
                'cov': y_train_gamma_auxiliar.tolist(),
                'phi': phi,
                'sigma': sigma,
                # 'bernoulli': ber,
                'p_seco': ber_seco,
                'p_transitorio': ber_transitorio,
                'p_lluvioso': ber_lluvioso,
                'beta3': beta3,
                'rho': rho,
                'error': error
            })
    return filas

# Número de núcleos para paralelización
n_cores = 10
# Número de simulaciones por núcleo
nsimul = 50

def main():
    from multiprocessing import Pool

    seeds = np.arange(1001, 1001 + n_cores)
    print(f"Ejecutando con {n_cores} núcleos...")

    with Pool(processes = n_cores) as pool:
        resultados = pool.starmap(
            corrida_simulacion,
            [
                (real, cov_original, m, nsites, meses, dist_mat, seeds[i])
                for i in range(n_cores)
            ]
        )

    filas_totales = [fila for sublist in resultados for fila in sublist]
    return filas_totales

# pd.DataFrame(filas_totales).to_parquet('resultados_abc_los_santos3.parquet')

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    inicio = datetime.now() 
    print("Hora inicial:", inicio)
    
    print(str(nsites) + ' locaciones')
    print(str(m) + ' dias de datos de precipitacion para entrenamiento')
    print("Cantidad de variantes de modelo: 6")
    print("Cantidad de variantes de covariables: 7")
    print("Cantidad de combinaciones de modelos y covariables: 42")
    print(f"Dimensiones esperadas del dataframe de resultados: {6*7*n_cores*nsimul} filas x {11} columnas")

    filas_totales = main()
    df = pd.DataFrame(filas_totales)
    df.to_parquet("resultados_prueba1.parquet")

    fin = datetime.now()
    print("Hora final:", fin)

    duracion = fin - inicio
    horas, resto = divmod(duracion.total_seconds(), 3600)
    minutos, segundos = divmod(resto, 60)

    print(f"Duración total: {int(horas)} horas, {int(minutos)} minutos, {int(segundos)} segundos")
    print("Dimensiones guardadas: ", df.shape)