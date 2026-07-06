#
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma, norm, multivariate_normal, wasserstein_distance
from scipy.special import gamma as xgamma, factorial
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
import os

# Reproducibilidad
rng = np.random.default_rng(123)

n_sitios = 10
n_obs = 120
n = n_sitios * n_obs

# Coordenadas de cada sitio
lon_sitios = rng.uniform(-1, -0.5, size=n_sitios)
lat_sitios = rng.uniform(0.5, 1, size=n_sitios)
oscilacion_base = np.sin(np.linspace(0, 2*np.pi, n_obs))

X = pd.DataFrame({
    "intercepto": np.ones(n),
    "dummy": rng.integers(0, 2, size=n),
    "lon": np.repeat(lon_sitios, n_obs),
    "lat": np.repeat(lat_sitios, n_obs),
    "oscilacion": np.tile(oscilacion_base, n_sitios)
})

# Betas verdaderos entre -2 y 2
beta = [3, 0.2, -0.05, 0.05, 2]

# Datos sintéticos ("reales")
Y = np.exp(X.values @ beta) #+ rng.normal(0, 0.1, size=n))

print("Betas verdaderos:")
print(beta)

print("Primeras observaciones:")
print(Y[:120])

def simular_betas_abc(n_sim, n_betas, seed=1234):
    rng = np.random.default_rng(seed)
    # betas = rng.uniform(low=-10, high=10, size=(n_sim, n_betas))
    betas = rng.normal(loc=0, scale=10, size=(n_sim, n_betas))
    return betas

def mse(y_real, y_sim):
    return np.mean((y_real - y_sim) ** 2)

def abc_betas(X, Y, n_sim=10_000, seed=1234):
    rng = np.random.default_rng(seed)
    
    X_mat = X.values
    n_betas = X_mat.shape[1]
    
    betas_sim = simular_betas_abc(n_sim, n_betas, seed=seed)
    
    resultados = []
    
    for i in range(n_sim):
        beta_i = betas_sim[i]
        
        Y_sim = np.exp(X_mat @ beta_i)
        
        errorMSE = mse(Y, Y_sim)
        
        resultados.append({
            "sim": i,
            "errorMSE": errorMSE,
            **{f"beta_{j}": beta_i[j] for j in range(n_betas)}
        })
    
    return pd.DataFrame(resultados)

def abc_betas_log(X, Y, n_sim=10_000, seed=1234):
    rng = np.random.default_rng(seed)
    
    X_mat = X.values
    logY = np.log(Y)
    n_betas = X_mat.shape[1]

    betas_sim = simular_betas_abc(n_sim, n_betas, seed=seed)
    
    resultados = []
    
    for i in range(n_sim):
        beta_i = betas_sim[i]
        
        logY_sim = X_mat @ beta_i
        
        errorMSE = np.mean((logY - logY_sim) ** 2)
        
        resultados.append({
            "sim": i,
            "errorMSE": errorMSE,
            **{f"beta_{j}": beta_i[j] for j in range(n_betas)}
        })
    
    return pd.DataFrame(resultados)

def mejores_betas_abc(resultados, p=0.01):
    umbral = resultados["errorMSE"].quantile(p)
    mejores = resultados[resultados["errorMSE"] <= umbral]
    mejores = mejores.sort_values("errorMSE")
    return mejores

resultados_abc = abc_betas_log(
    X,
    Y,
    n_sim=1000000,
    seed=1234
)

mejores = mejores_betas_abc(resultados_abc, p=0.001)

print(mejores.head())

mejor = mejores.iloc[0]

beta_estimado = mejor[[f"beta_{j}" for j in range(X.shape[1])]]

resumen_betas = mejores[
    [f"beta_{j}" for j in range(X.shape[1])]
].agg(["mean", "std", "min", "max"])

resumen_betas.columns = X.columns

print(resumen_betas)