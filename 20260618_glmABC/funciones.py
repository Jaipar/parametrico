# Funciones para el modelo Approximate Bayesian Computation (ABC)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma, norm, multivariate_normal, wasserstein_distance
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from datetime import datetime
import os

def importar_ENSO():
    url = "https://psl.noaa.gov/data/correlation/oni.data"
    
    df = pd.read_csv(
        url,
        sep=r"\s+",
        skiprows=1,
        engine="python",
        on_bad_lines="skip"
    )

    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]

    df.columns = [
        "year","DJF","JFM","FMA","MAM","AMJ","MJJ",
        "JJA","JAS","ASO","SON","OND","NDJ"
    ]

    df = df.melt(
        id_vars="year",
        var_name="season",
        value_name="ONI"
    )

    df["ONI"] = pd.to_numeric(df["ONI"], errors="coerce")
    df = df.dropna(subset=["ONI"])
    df = df[df["ONI"] != -99.90]

    mapa_mes = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }

    df["month"] = df["season"].map(mapa_mes)
    df["year"] = df["year"].astype(int)

    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    # Condiciones para fenómeno Niño o Niña, mayores o menores a +- 0.5
    condiciones = np.where(df["ONI"] >= 0.5, 1,
                   np.where(df["ONI"] <= -0.5, -1, 0))
    df["signal"] = condiciones

    # Condiciones para fenómeno Niño o Niña, cinco o más períodos sobrelapados
    df["grupo"] = (df["signal"] != df["signal"].shift()).cumsum()

    # Contar longitud de cada racha
    tamaños = df.groupby("grupo")["signal"].transform("size")

    # Clasificación final
    def clasificar(row):
        if row["signal"] == 1 and tamaños[row.name] >= 5:
            return "El Niño"
        elif row["signal"] == -1 and tamaños[row.name] >= 5:
            return "La Niña"
        else:
            return "Neutral"

    df["ENSO"] = df.apply(clasificar, axis=1)
    df = df.drop(columns=["signal", "grupo"])

    return df

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

# Métrica de error Sliced Wasserstein Distance
# Multiples proyecciones a 1D y posteriormente se promedian las distancias de Wasserstein para cada proyección
def sliced_wasserstein(X, Y, n_projections=200, seed=1234):
    rng_seed = np.random.default_rng(seed)
    d = X.shape[1]
    distances = []

    for _ in range(n_projections):
        theta = rng_seed.normal(size=d)
        norm = np.linalg.norm(theta)
        theta /= norm

        X_proj = X @ theta
        Y_proj = Y @ theta

        distances.append(wasserstein_distance(X_proj, Y_proj))

    return np.mean(distances)

# Métrica de error cuatrático medio (MSE)
def mse(X, Y):
    return np.mean((X - Y) ** 2)

# Métrica de error medio absoluto (MAE)
def mae(X, Y):
    return np.mean(np.abs(X - Y))

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

    # Agregar términos de la serie de Fourier para capturar estacionalidad mensual
    datos_region = pd.concat([datos_region, fourier_mensual(datos_region['month'])], axis=1)

    return datos_region

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

    # Importar datos de NOAA (ONI index)
    enso_noaa = importar_ENSO()[["year", "month", "ENSO"]]

    # Incluir ENSO
    datos_region = datos_region.merge(enso_noaa, on=["year", "month"], how="left")
    datos_region['ENSO'] = pd.Categorical(datos_region['ENSO'], categories=['Neutral', 'El Niño', 'La Niña'], ordered=False)

    return datos_region

def dividir_train_test(datos, ubic, nsites, fecha_corte, frac=0.75, seed=1234):
    # Seleccionar aleatoriamente un subconjunto de ubicaciones para entrenamiento    
    n_ubic_train = int(frac * nsites)
    ubic_train = ubic.sample(n=n_ubic_train, random_state=seed)
    
    # Convertir a set de tuplas para lookup rápido
    ubic_train_set = set(zip(ubic_train['lon'], ubic_train['lat']))

    # Crear máscaras para filtrado
    mask_time_train = datos['date'] <= fecha_corte
    mask_ubic_train = datos.apply(lambda row: (row['lon'], row['lat']) in ubic_train_set, axis=1)

    # Separar datos de entrenamiento y prueba
    datos_train = datos[mask_time_train & mask_ubic_train]
    datos_test = datos[~(mask_time_train & mask_ubic_train)]

    # Cantidad de meses (filas) para entrenamiento por sitio
    m_train = int(len(datos_train)/n_ubic_train)

    return datos_train, datos_test, m_train

def parametros_ubicaciones(datos):
    # Combinaciones únicas de latitud y longitud
    ubic = (
        datos[['lon','lat','elevation']]
        .drop_duplicates()
        .sort_values(['lat','lon'], ascending=[False,True])
        .reset_index(drop=True)
    )
    # Número de ubicaciones únicas
    nsites = ubic.shape[0]

    # Matriz de distancias entre ubicaciones
    dist_mat = squareform(pdist(ubic[['lon','lat']])) # Dimensiones: nsites x nsites

    return ubic, nsites, dist_mat

columnas_polinomicas = ['lat', 'lon']

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

columnas_no_polinomicas = ['elevation', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'ENSO']

columnas_regresion_lineal = ['lat', 'lon', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'ENSO']

def procesamiento_design_matrix(datos, poly, scaler):
    # Terminos polinomiales de latitud y longitud
    X_poly = poly.transform(datos[columnas_polinomicas])
    poly_cols = poly.get_feature_names_out(columnas_polinomicas)

    X_poly = pd.DataFrame(X_poly, columns=poly_cols, index=datos.index)
    X_poly = pd.DataFrame(scaler.transform(X_poly), columns=poly_cols, index=datos.index)

    # Terminos sin transformaciones polinomiales
    X_nopoly = datos[columnas_no_polinomicas]

    # Combinar términos polinomiales y no polinomiales
    X_full = pd.concat([X_poly, X_nopoly], axis=1)

    # Selección final
    X_full = X_full[columnas_regresion_lineal]

    # Tratamiento de categorías ENSO. Incluir dummies
    if 'ENSO' in X_full.columns:
        dummies_enso = pd.get_dummies(X_full['ENSO'], drop_first=True)
        X_full = pd.concat([X_full, dummies_enso], axis=1)
        X_full = X_full.drop(columns=['ENSO'])

    return X_full

def calibrar_modelo_GLM(train, print_summary=False):
    poly, scaler = preprocesamiento_escalar_datos(train)
    X_regresion_lineal = procesamiento_design_matrix(train, poly, scaler)

    Y_plus = train['chirps'] + 1
    X_sm = sm.add_constant(X_regresion_lineal)
    model_GLM = sm.GLM(Y_plus, X_sm, family=sm.families.Gamma(link=sm.families.links.log())).fit()
    if print_summary:
        print(model_GLM.summary())

    return model_GLM

def GLM_predicciones_residuos_train(train, model_GLM):
    Y_plus = train['chirps'] + 1
    Y_plus_pred = model_GLM.fittedvalues
    RatioY = Y_plus/Y_plus_pred

    result_df = pd.DataFrame({
        "Y_plus": Y_plus,
        "Y_plus_pred": Y_plus_pred,
        "exp_residuos": RatioY,
    }, index=train.index)

    return pd.concat([train, result_df], axis=1)

def GLM_predicciones_residuos_test(test, poly, scaler, model_GLM):
    X_regresion_lineal = procesamiento_design_matrix(test, poly, scaler)
    X_sm = sm.add_constant(X_regresion_lineal)

    Y_plus = test['chirps'] + 1
    Y_plus_pred = model_GLM.predict(X_sm).values
    RatioY = Y_plus/Y_plus_pred

    result_df = pd.DataFrame({
        "Y_plus": Y_plus,
        "Y_plus_pred": Y_plus_pred,
        "exp_residuos": RatioY,
    }, index=test.index)

    return pd.concat([test, result_df], axis=1)

# Cálculo de proceso AR(1)
def simular_logAR1(n, phi, sigma, nsites=1):
    observaciones = np.zeros((n, nsites))
    ce = -(sigma**2) / (2 * (1 - phi**2))
    observaciones[0, :] = np.exp(np.random.normal(ce, sigma / np.sqrt(1 - phi**2), size=nsites))
    for t in range(1, n):
        observaciones[t, :] = np.exp( (1 - phi) * ce + phi * np.log(observaciones[t-1,:]) + np.random.normal(0, sigma, size=nsites) )
        # log(observaciones[t, :]) = (1-phi)*ce + phi*log(observaciones[t-1,:]) + np.random.normal(0, sigma)
    return observaciones

# Cálculo de proceso AR(12)
def simular_logAR12(n, phi, sigma, nsites=1):
    observaciones = np.zeros((n, nsites))
    ce = -(sigma**2) / (2 * (1 - phi**2))
    for t in range(0, 12):
        observaciones[t, :] = np.exp(np.random.normal(ce, sigma / np.sqrt(1 - phi**2), size=nsites))
    for t in range(12, n):
        observaciones[t, :] = np.exp( (1 - phi) * ce + phi * np.log(observaciones[t-12,:]) + np.random.normal(0, sigma, size=nsites) )
        # log(observaciones[t, :]) = (1-phi)*ce + phi*log(observaciones[t-1,:]) + np.random.normal(0, sigma)
    return observaciones

# Calculo de proceso X_3t(s) con cópula subyacente C_X_3 y distribución marginal F_3 con media 1 y cola regular (Gamma Inversa)
def simular_X3(n, rho, beta3, nsites, dist_mat):
  Sigma = np.exp(-dist_mat / rho)
  Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados correlacionados (dimensiones: ntime x nsites)
  Gamma = gamma.ppf(norm.cdf(Gauss), a=beta3, scale=1)
  X3 = (beta3 - 1)/Gamma
  return X3

# Previas para parámetros del modelo (phi, sigma, beta3, rho)
def model_prior(rho_upper_range):
    # Phi1: Grado de autocorrelación temporal AR(1) para X_2t(s)
    y_train_phi12_auxiliar = np.random.uniform(-0.85,0.85)
    # Sigma1: Desviación estándar del proceso logAR(1) para X_2t(s)
    y_train_sigma12_auxiliar = np.random.uniform(0,3)
    # Phi: Grado de autocorrelación temporal AR(1) para X_2t(s)
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85)
    # Sigma: Desviación estándar del proceso logAR(1) para X_2t(s)
    y_train_sigma_auxiliar = np.random.uniform(0,3)
    # Beta3: Parámetro de la distribución marginal de X_3t(s) Gamma Inversa
    y_train_beta3_auxiliar = np.random.uniform(2,7.5)
    # Rho: Parámetro de la correlación espacial de X_3t(s)
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range)

    y_train_auxiliar = [y_train_phi12_auxiliar,
                        y_train_sigma12_auxiliar,
                        y_train_phi_auxiliar,
                        y_train_sigma_auxiliar,
                        y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
    
    previas = np.array(y_train_auxiliar)
    return(previas)

def comprobar_dimensiones_ubic(ubic, datos_pivot):
    cols = datos_pivot.columns
    
    check = (
        (cols.get_level_values(0).values == ubic['lat'].values) &
        (cols.get_level_values(1).values == ubic['lon'].values)
    )

    return check.all()

def corrida_simulacion(real, m, nsites, dist_mat, nsimul, seed):
    print(f"PID {os.getpid()} seed {seed}")
    print("Cantidad de simulaciones: ", nsimul)

    filas = []
    np.random.seed(seed)

    # Rango superior para rho (parámetro de la función de correlación espacial)
    rho_upper_range = 2*np.max(dist_mat)

    for _ in range(nsimul):
        phi12, sigma12, phi, sigma, beta3, rho = model_prior(rho_upper_range)

        # Procesos comunes para todos los sitios
        X1_auxiliar_fijo = simular_logAR12(m, phi12, sigma12).reshape(m, 1) # m x 1
        X2_auxiliar_fijo = simular_logAR1(m, phi, sigma).reshape(m, 1) # m x 1 

        # Procesos específicos para cada sitio (forma matricial)
        X1_auxiliar_completo = simular_logAR12(m, phi12, sigma12, nsites) # m x nsites
        X2_auxiliar_completo = simular_logAR1(m, phi, sigma, nsites) # m x nsites
        X3_auxiliar_completo = simular_X3(m, rho, beta3, nsites, dist_mat) # m x nsites

        # Modelos
        X_train_auxiliar_D3 = X1_auxiliar_fijo * X3_auxiliar_completo
        X_train_auxiliar_D4 = X1_auxiliar_completo * X3_auxiliar_completo
        X_train_auxiliar_D5 = X1_auxiliar_fijo * X2_auxiliar_fijo * X3_auxiliar_completo
        X_train_auxiliar_D6 = X1_auxiliar_completo * X2_auxiliar_fijo * X3_auxiliar_completo
        X_train_auxiliar_D7 = X1_auxiliar_fijo * X2_auxiliar_completo * X3_auxiliar_completo
        X_train_auxiliar_D8 = X1_auxiliar_completo * X2_auxiliar_completo * X3_auxiliar_completo

        # 6 variantes de modelos
        simulaciones = [X_train_auxiliar_D3,X_train_auxiliar_D4,X_train_auxiliar_D5,
                        X_train_auxiliar_D6,X_train_auxiliar_D7,X_train_auxiliar_D8]

        # Modelo solo con intercepto (sin covariables)
        for s in range(len(simulaciones)):
            errorMSE = mse(real, simulaciones[s])
            errorSWD = sliced_wasserstein(real, simulaciones[s])
            # 9 columnas
            filas.append({
                'modelo': 'D' + str(s + 3),
                'covariables': 'M1',
                'phi12': phi12,
                'sigma12': sigma12,
                'phi': phi,
                'sigma': sigma,
                'beta3': beta3,
                'rho': rho,
                'errorMSE': errorMSE,
                'errorSWD': errorSWD
            })

    return filas