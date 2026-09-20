import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gamma as gamma_function
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma, norm, multivariate_normal
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

# ============================================================
# 0. FUNCIONES AUXILIARES
# ============================================================

# Estacionalidad mensual mediante serie de Fourier
def fourier_mensual(mes):
    fourier = np.column_stack([
        np.sin(2 * np.pi * mes / 12).round(3),
        np.cos(2 * np.pi * mes / 12).round(3),
        np.sin(4 * np.pi * mes / 12).round(3),
        np.cos(4 * np.pi * mes / 12).round(3)
    ])
    fourier_df = pd.DataFrame(fourier, columns=['sin_1', 'cos_1', 'sin_2', 'cos_2'])
    return fourier_df

# Preparar datos para una región específica, incluyendo precipitación mensual y covariables espaciales
def preparar_datos_region(datos, region):
    # Filtrar por región
    datos_region = datos.loc[datos['region'] == region].copy()

    datos_region['lon'] = datos_region['lon'].round(3)
    datos_region['lat'] = datos_region['lat'].round(3)

    # Preparar dataframe con datos mensuales de precipitación y covariables espaciales (latitud, longitud, elevación)
    datos_region['date'] = pd.to_datetime(datos_region['date'])
    datos_region = datos_region.sort_values('date').reset_index(drop=True)
    datos_region['year_month'] = datos_region['date'].dt.to_period('M')
    datos_region = (
        datos_region.
        groupby(['lat', 'lon', 'year_month']).
        agg({
            'chirps': 'sum',
            'elevation': 'first',
            'date': 'first'}).
        reset_index().
        copy()
    )
    # Columna con número de mes (1 - 12)
    datos_region["month"] = datos_region["date"].dt.month
    # Columna con número de año
    datos_region["year"] = datos_region["date"].dt.year

    # Agregar términos de la serie de Fourier para capturar estacionalidad mensual
    datos_region = pd.concat([datos_region, fourier_mensual(datos_region['month'])], axis=1)

    return datos_region

# Obtener conjunto de ubicaciones únicas
def identificar_ubic(datos):
    # Combinaciones únicas de latitud y longitud
    ubic = (
        datos[['lon','lat']]
        .drop_duplicates()
        .sort_values(['lat','lon'], ascending=[False,True])
        .reset_index(drop=True)
    )
    return ubic

def calcular_matriz_distancias(ubic):
    # Matriz de distancias entre ubicaciones
    dist_mat = squareform(pdist(ubic[['lon','lat']])) # Dimensiones: nsites x nsites
    return dist_mat

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_train_test(datos, fecha_corte, frac=0.75, seed=1234):
    ubic_total = identificar_ubic(datos)
    nsites_total = ubic_total.shape[0]

    # Seleccionar aleatoriamente un subconjunto de ubicaciones para entrenamiento
    nsites_train = int(frac * nsites_total)
    nsites_test = nsites_total - nsites_train
    ubic_train = ubic_total.sample(n=nsites_train, random_state=seed)

    # Identificar ubicaciones de testing: ubicaciones no seleccionadas en training
    indice_ubic_total = pd.MultiIndex.from_frame(ubic_total[['lon', 'lat']])
    indice_ubic_train = pd.MultiIndex.from_frame(ubic_train[['lon', 'lat']])
    mask_ubic_testing = ~indice_ubic_total.isin(indice_ubic_train)
    ubic_test = ubic_total[mask_ubic_testing]

    # Máscaras sobre las filas de datos
    indice_datos = pd.MultiIndex.from_frame(datos[['lon', 'lat']])
    mask_ubic_train = indice_datos.isin(indice_ubic_train)
    mask_time = datos['date'] <= fecha_corte

    # Separar datos de entrenamiento y prueba
    datos_train = datos[mask_ubic_train & mask_time]
    datos_test = datos[~mask_time]

    # Cantidad de meses (filas) por sitio
    m_train = int(len(datos_train) / nsites_train)
    m_test = int(len(datos_test) / nsites_total)

    return datos_train, datos_test, nsites_train, nsites_test, nsites_total, ubic_train, ubic_test, ubic_total, m_train, m_test

# Constructor de objetos para calcular términos polinomiales y estandarización de datos
def preprocesamiento_escalar_datos(train):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = StandardScaler()

    # Constructor de términos polinomiales
    X_poly = poly.fit_transform(train[columnas_polinomicas])
    poly_cols = poly.get_feature_names_out(columnas_polinomicas)
    X_poly = pd.DataFrame(X_poly, columns=poly_cols, index=train.index)

    # Constructor de estandarización
    scaler.fit(X_poly)

    return poly, scaler

columnas_polinomicas = ['lat', 'lon']
columnas_no_polinomicas = ['elevation', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'month']

# Construir matriz de diseño
def procesamiento_matriz_diseno(datos, columnas_matriz_diseno, poly, scaler):
    # Terminos polinomiales de latitud y longitud
    X_poly = poly.transform(datos[columnas_polinomicas])
    poly_cols = poly.get_feature_names_out(columnas_polinomicas)

    X_poly = pd.DataFrame(X_poly, columns=poly_cols, index=datos.index)
    X_poly = pd.DataFrame(scaler.transform(X_poly), columns=poly_cols, index=datos.index)

    # Terminos sin transformaciones polinomiales
    X_nopoly = datos[columnas_no_polinomicas]

    # Combinar términos polinomiales y no polinomiales
    X_full = pd.concat([X_poly, X_nopoly], axis=1)

    # Selección de columnas para la matriz diseño
    X_design = X_full[columnas_matriz_diseno]

    # Tratamiento de variables categóricas
    for col in ["ENSO", "month"]:
        if col in columnas_matriz_diseno:
            dummies = pd.get_dummies(X_design[col], prefix=col, drop_first=True)
            X_design = pd.concat([X_design.drop(columns=[col]), dummies], axis=1)

    # Incluir constante
    X_design = sm.add_constant(X_design)

    return X_design

# X_1t(s) con distribución marginal Lognormal
def simular_X1_lognormal(n, delta, nsites=1):
    lognormal = np.random.lognormal(mean=0, sigma=delta, size=(n, nsites))
    return lognormal / np.exp(delta**2 / 2)

# X_1t(s) con distribución marginal Weibull
def simular_X1_weibull(n, kappa, nsites=1):
    weibull = np.random.weibull(kappa, size=(n, nsites))
    return weibull / gamma_function(1 + 1/kappa)

# Calculo de proceso X_3t(s) con cópula subyacente C_X_3 y distribución marginal Gamma Inversa
def simular_X3(n, betax, rho, nsites, dist_mat):
    Sigma = np.exp(-dist_mat / rho)
    Gauss = np.random.multivariate_normal(mean=np.zeros(nsites), cov=Sigma, size=n)
    Gamma = gamma.ppf(norm.cdf(Gauss), a=betax, scale=1)
    X3 = (betax - 1) / Gamma
    if not np.all(np.isfinite(X3)):
        print("PROBLEMA X3")
        print("beta:", betax, "rho:", rho)
        print("Gauss min/max:", Gauss.min(), Gauss.max())
        print("Gamma min/max:", np.nanmin(Gamma), np.nanmax(Gamma))
        print("X3 NaN:", np.isnan(X3).sum())
        print("X3 Inf:", np.isinf(X3).sum())
    return X3