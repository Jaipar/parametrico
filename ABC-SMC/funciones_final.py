# Funciones para el modelo Approximate Bayesian Computation (ABC)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from multiprocessing import get_context
from scipy.linalg import cholesky, solve_triangular
from scipy.special import gammaincinv, logsumexp, ndtr
from scipy.special import gamma as gamma_function
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import gamma, norm, multivariate_normal, wasserstein_distance
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from datetime import datetime
import os

# Incluir columnas de fenómeno ENSO (El Niño, La Niña, Neutral)
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

    # Identificar ONI mayores o menores a +- 0.5
    df["signal"] = np.where(df["ONI"] >= 0.5, 1, np.where(df["ONI"] <= -0.5, -1, 0))
    # Identificar rachas de señales consecutivas iguales
    df["grupo"] = (df["signal"] != df["signal"].shift()).cumsum()

    # Contar longitud de cada racha
    tamaños = df.groupby("grupo")["signal"].transform("size")

    # Condiciones para fenómeno Niño o Niña: Cinco o más períodos sobrelapados con ONI >= 0.5 o ONI <= -0.5 
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

# Métrica de error cuatrático medio (MSE)
def mse(X, Y):
    return np.mean((X - Y) ** 2)

# Métrica de error medio absoluto (MAE)
def mae(X, Y):
    return np.mean(np.abs(X - Y))

# Métrica de error cuadrático cuantílico promedio (ECCP)
def error_cuadratico_cuantilico(X, Y):
    c = 10 # Número de cuantiles en cada extremo
    cuantiles = np.concatenate([np.arange(1, c + 1),np.arange(100 - c, 100)]) / 100

    q_X = np.quantile(X, cuantiles, axis=0)
    q_Y = np.quantile(Y, cuantiles, axis=0)

    eccp = mse(q_X, q_Y)
    return eccp

# Métrica de error absoluto cuantílico promedio (EACP)
def error_absoluto_cuantilico(X, Y):
    c = 10 # Número de cuantiles en cada extremo
    cuantiles = np.concatenate([np.arange(1, c + 1),np.arange(100 - c, 100)]) / 100

    q_X = np.quantile(X, cuantiles, axis=0)
    q_Y = np.quantile(Y, cuantiles, axis=0)

    eacp = mae(q_X, q_Y)
    return eacp


######## Funciones por borrar
def preparar_metricas_colas(X, meses, cuantil_umbral=0.90):
    p_cola = np.linspace(cuantil_umbral, 0.99, 10)
    cuantiles_cola_mes = np.empty((12, 10), dtype=float)
    umbrales_ubicaciones_mes = np.empty((12, X.shape[1]), dtype=float)

    for mes in range(1, 12+1):
        condicion = meses == mes
        X_mes = X[condicion]

        cuantiles_cola_mes[mes - 1] = np.quantile(X_mes.ravel(),p_cola)
        umbrales_ubicaciones_mes[mes - 1] = np.quantile(X_mes, cuantil_umbral, axis=0)

    proporciones_exceder_umbral = np.mean(X > umbrales_ubicaciones_mes[meses - 1], axis=1)

    # Revisar
    return p_cola, cuantiles_cola_mes, umbrales_ubicaciones_mes, proporciones_exceder_umbral

def error_cola_mensual(Y, meses, p_cola, cuantiles_cola_mes):
    errores_mensuales = np.empty(12, dtype=float)

    for mes in range(1, 12+1):
        Y_mes = Y[meses == mes].ravel()
        Y_cuantiles_mes = np.quantile(Y_mes, p_cola)
        errores_mensuales[mes - 1] = mse(cuantiles_cola_mes[mes - 1], Y_cuantiles_mes)

    return np.mean(errores_mensuales)

def error_exceder_umbral(Y, meses, umbrales_ubicaciones_mes, proporciones_exceder_umbral):
    Y_proporciones_exceder_umbral = np.mean(Y > umbrales_ubicaciones_mes[meses - 1], axis=1)
    errores_mensuales = np.empty(12, dtype=float)

    for mes in range(1, 12+1):
        condicion = meses == mes
        X_mes = np.sort(proporciones_exceder_umbral[condicion])
        Y_mes = np.sort(Y_proporciones_exceder_umbral[condicion])
        errores_mensuales[mes - 1] = mse(X_mes, Y_mes)

    return np.mean(errores_mensuales)
######### Fin funciones por borrar

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

    # Importar datos de NOAA (ONI index)
    enso_noaa = importar_ENSO()[["year", "month", "ENSO"]]

    # Incluir ENSO
    datos_region = datos_region.merge(enso_noaa, on=["year", "month"], how="left")
    datos_region['ENSO'] = pd.Categorical(datos_region['ENSO'], categories=['Neutral', 'El Niño', 'La Niña'], ordered=False)

    return datos_region

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_train_test(datos, ubic, nsites, fecha_corte, frac=0.75, seed=1234):
    # Seleccionar aleatoriamente un subconjunto de ubicaciones para entrenamiento    
    n_ubic_train = int(frac * nsites)
    ubic_train = ubic.sample(n=n_ubic_train, random_state=seed)
    
    # Crear máscaras para filtrado
    mask_time_train = datos['date'] <= fecha_corte
    indice_ubicaciones = pd.MultiIndex.from_frame(datos[['lon', 'lat']])
    indice_training = pd.MultiIndex.from_frame(ubic_train[['lon', 'lat']])
    mask_ubic_train = indice_ubicaciones.isin(indice_training)

    # Separar datos de entrenamiento y prueba
    datos_train = datos[mask_time_train & mask_ubic_train]
    datos_test = datos[~(mask_time_train & mask_ubic_train)]

    # Cantidad de meses (filas) para entrenamiento por sitio
    m_train = int(len(datos_train)/n_ubic_train)

    return datos_train, datos_test, m_train

# Obtener conjunto de ubicaciones únicas, número de sitios y matriz de distancias entre ubicaciones
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
columnas_no_polinomicas = ['elevation', 'sin_1', 'cos_1', 'sin_2', 'cos_2', 'ENSO', 'month']
# Columnas seleccionadas para la matriz de diseño
columnas_matriz_diseno = ['lat', 'lon', 'month']

# Construir matriz de diseño
def procesamiento_matriz_diseno(datos, poly, scaler):
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

# Modelo GLM sobre Y+1
def calibrar_modelo_GLM(train, print_summary=False, poly=None, scaler=None):
    X_design = procesamiento_matriz_diseno(train, poly, scaler)
    Y_plus = train['chirps'] + 1

    model_GLM = sm.GLM(Y_plus, X_design, family=sm.families.Gamma(link=sm.families.links.log())).fit()

    if print_summary:
        print(model_GLM.summary())

    return model_GLM

# Incluir columnas de predicciones y residuos del modelo GLM para el conjunto de entrenamiento
def GLM_predicciones_residuos_train(train, model_GLM):
    Y_plus = train['chirps'] + 1
    Y_plus_pred = model_GLM.fittedvalues
    Y_resid = model_GLM.resid_response
    Y_ratio = Y_plus/Y_plus_pred
    
    result_df = pd.DataFrame({
        "Y_plus": Y_plus,
        "Y_plus_pred": Y_plus_pred,
        "Y_resid": Y_resid,
        "Y_ratio": Y_ratio,
    })

    return pd.concat([train, result_df], axis=1)

# Incluir columnas de predicciones y residuos del modelo GLM para el conjunto de prueba
def GLM_predicciones_residuos_test(test, poly, scaler, model_GLM):
    X_design = procesamiento_matriz_diseno(test, poly, scaler)

    Y_plus = test['chirps'] + 1
    Y_plus_pred = model_GLM.predict(X_design).values
    Y_resid = model_GLM.resid_response
    Y_ratio = Y_plus/Y_plus_pred

    result_df = pd.DataFrame({
        "Y_plus": Y_plus,
        "Y_plus_pred": Y_plus_pred,
        "Y_resid": Y_resid,
        "Y_ratio": Y_ratio
    })

    return pd.concat([test, result_df], axis=1)

# Cálculo del proceso X_1t(s) con distribución marginal Lognormal
def simular_X1(n, delta, nsites=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    lognormal = rng.lognormal(mean=0, sigma=delta, size=(n, nsites))
    return lognormal / np.exp(delta**2 / 2)

# Calculo de proceso X_1t(s) con distribución marginal Weibull
def simular_X1(n, k, nsites=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    weibull = rng.weibull(k, size=(n, nsites))
    return weibull / gamma_function(1 + 1/k)

# Calculo de proceso X_3t(s) con cópula subyacente C_X_3 y distribución marginal Gamma Inversa
def simular_X3(n, rho, beta3, nsites, dist_mat, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Sigma = np.exp(-dist_mat / rho)
    Gauss = rng.multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)
    Gamma = gamma.ppf(norm.cdf(Gauss), a=beta3, scale=1)
    return (beta3 - 1) / Gamma

# Distribución normal multivariada para distribucion previa
def model_prior_mvnormal(n_muestras, vector_mu, cov_matriz, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    muestras_previa = rng.multivariate_normal(mean=vector_mu, cov=cov_matriz, size=n_muestras)
    return muestras_previa

# Comprobar si las dimensiones de ubicaciones coincidan en el mismo orden (después de pivotear)
def comprobar_dimensiones_ubic(ubic, datos_pivot):
    cols = datos_pivot.columns
    check_lat = cols.get_level_values(0).values == ubic['lat'].values
    check_lon = cols.get_level_values(1).values == ubic['lon'].values
    return (check_lat & check_lon).all()

# Construir parámetros de la distribución previa del modelo ABC-SMC mediante GLM
def construir_parametros_previa_glm(GLM0, n_extra=3, c=10):
    gamma_hat = GLM0.params.to_numpy()
    cov_gamma = GLM0.cov_params().to_numpy()
    cov_gamma = (c**2) * cov_gamma

    ngammas = len(gamma_hat)
    nparametros = ngammas + n_extra

    # Coeficientes de alpha(s,t) (gammas)
    mu_0 = np.zeros(nparametros)
    mu_0[:ngammas] = gamma_hat

    cov_matriz_0 = np.zeros((nparametros, nparametros))
    cov_matriz_0[:ngammas, :ngammas] = cov_gamma
    cov_matriz_0[ngammas:, ngammas:] = np.eye(n_extra)

    return mu_0, cov_matriz_0

# Transformar valores del vector theta en las escalas requeridas para la generación aleatoria (v1)
def transformar_theta_v1(theta, ngammas, rho_upper_range):
    X1_param_max_range = 3
    X3_param_min_range = 2
    X3_param_max_range = 150

    gamma_params = theta[:ngammas]
    X1_param = X1_param_max_range * norm.cdf(theta[ngammas])
    X3_param = X3_param_min_range + (X3_param_max_range - X3_param_min_range) * norm.cdf(theta[ngammas + 1])
    rho = rho_upper_range * norm.cdf(theta[ngammas + 2])
    return gamma_params, X1_param, X3_param, rho

# Transformar valores del vector theta en las escalas requeridas para la generación aleatoria (v2)
def transformar_theta_v2(theta, ngammas, rho_upper_range):
    X3_param_min_range = 2
    X3_param_max_range = 150

    gamma_params = theta[:ngammas]
    X3_param = X3_param_min_range + (X3_param_max_range - X3_param_min_range) * norm.cdf(theta[ngammas + 0])
    rho = rho_upper_range * norm.cdf(theta[ngammas + 1])
    return gamma_params, X3_param, rho

# Simular precipitaciones (v1)
def simular_precipitacion_v1(theta, m, nsites, dist_mat, design_mat, ngammas, rho_upper_range, rng=None):
    gamma_params, delta, beta3, rho = transformar_theta_v1(theta, ngammas, rho_upper_range)

    alpha_long = np.exp(design_mat @ gamma_params)
    alpha = alpha_long.reshape(nsites, m).T # m x nsites

    X1 = simular_X1(m, delta, 1, rng) # m x 1. Constante por ubicación
    X3 = simular_X3(m, rho, beta3, nsites, dist_mat, rng) # m x nsites
    Y_sim = alpha * X1 * X3 - 1 # m x nsites

    return Y_sim

# Simular precipitaciones (v2)
def simular_precipitacion_v2(theta, m, nsites, dist_mat, design_mat, ngammas, rho_upper_range, rng=None):
    gamma_params, beta3, rho = transformar_theta_v2(theta, ngammas, rho_upper_range)

    alpha_long = np.exp(design_mat @ gamma_params)
    alpha = alpha_long.reshape(nsites, m).T # m x nsites

    X3 = simular_X3(m, rho, beta3, nsites, dist_mat, rng) # m x nsites
    Y_sim = alpha * X3 - 1 # m x nsites

    return Y_sim

# Argumentos globales del ABC-SMC
_worker_X = None
_worker_m = None
_worker_nsites = None
_worker_dist_mat = None
_worker_design_mat = None
_worker_ngammas = None
_worker_rho_upper_range = None

# Inicializar los argumentos constantes del ABC-SMC.
def _inicializar_worker(X, m, nsites, dist_mat, design_mat, ngammas, rho_upper_range):
    global _worker_X
    global _worker_m
    global _worker_nsites
    global _worker_dist_mat
    global _worker_design_mat
    global _worker_ngammas
    global _worker_rho_upper_range

    _worker_X = X
    _worker_m = m
    _worker_nsites = nsites
    _worker_dist_mat = dist_mat
    _worker_design_mat = design_mat
    _worker_ngammas = ngammas
    _worker_rho_upper_range = rho_upper_range

def _evaluar_bloque_candidatos(argumentos):

    candidatos, seed_bloque = argumentos
    rng = np.random.default_rng(seed_bloque)

    # Crear un arreglo para almacenar los errores de cada candidato
    errores = np.full((len(candidatos), 3), np.inf)

    for i, theta in enumerate(candidatos):
        # Evitar advertencias durante la generación de simulaciones
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Y = simular_precipitacion_v1(
                theta,
                _worker_m,
                _worker_nsites,
                _worker_dist_mat,
                _worker_design_mat,
                _worker_ngammas,
                _worker_rho_upper_range,
                rng=rng
            )
        # Si por error hay errores invalidos, continuar con el siguiente candidato
        if not np.isfinite(Y).all():
            continue
        errores[i] = mse(_worker_X, Y), mae(_worker_X, Y), error_cuadratico_cuantilico(_worker_X, Y)

    return errores

def _evaluar_candidatos_paralelo(candidatos, n_cores, semilla_epoca, pool):
    n_bloques = min(n_cores, len(candidatos))
    bloques = np.array_split(candidatos, n_bloques)
    semillas_bloques = semilla_epoca.spawn(n_bloques)

    tareas = [
        (bloque, int(semilla.generate_state(1, dtype=np.uint64)[0]))
        for bloque, semilla in zip(bloques, semillas_bloques)
    ]

    if pool is None:
        resultados = [_evaluar_bloque_candidatos(tarea) for tarea in tareas]
    else:
        resultados = pool.map(_evaluar_bloque_candidatos, tareas)

    return np.concatenate(resultados, axis=0)

def proceso_abc_smc(
    # Argumentos constantes del proceso ABC-SMC
    real,
    m,
    nsites,
    dist_mat,
    design_mat,
    GLM0,
    # Argumentos de control del proceso ABC-SMC
    n_cores,
    n_epochs,
    n_simul,
    n_min_aceptados,
    seed
):
    print(f"PID {os.getpid()} seed {seed}")

    # Argumentos para distribución previa
    rho_upper_range = 5 * np.max(dist_mat)
    mu_0, cov_matriz_0 = construir_parametros_previa_glm(GLM0)
    ngammas = len(GLM0.params.values)
    nparametros = len(mu_0)

    # Secuencia de semillas
    secuencia_semillas = np.random.SeedSequence(seed)
    secuencia_principal, secuencia_workers = secuencia_semillas.spawn(2)
    rng = np.random.default_rng(secuencia_principal)
    semillas_epocas = secuencia_workers.spawn(n_epochs)

    # Inicializar parámetros globales del proceso ABC-SMC reutilizados por los pools
    argumentos_worker = (real, m, nsites, dist_mat, design_mat, ngammas, rho_upper_range)
    pool = None

    # Generar un pool de procesos para la evaluación paralela de candidatos
    if n_cores == 1:
        _inicializar_worker(*argumentos_worker)
    else:
        pool = get_context("spawn").Pool(
            processes=n_cores,
            initializer=_inicializar_worker,
            initargs=argumentos_worker,
        )

    # Máximo número de candidatos permitidos para alcanzar el mínimo de aceptados
    n_max_candidatos = n_simul * 10

    # Primera poblacion de previas
    poblacion = model_prior_mvnormal(n_simul, mu_0, cov_matriz_0, rng=rng)
    pesos = np.ones(n_simul) / n_simul

    epsilon_epoch = np.inf

    try:
        for epoc in range(n_epochs):
            # Calcular kernel para las perturbaciones
            cov_kernel = calcular_cov_kernel(poblacion, pesos, nparametros, epoc)

            candidatos_bloques = []
            errores_bloques = []
            distancias_bloques = []
            total_candidatos = 0
            total_aceptados = 0

            while total_aceptados < n_min_aceptados:
                candidatos = generar_candidatos(n_simul, poblacion, cov_kernel, pesos, nparametros, epoc, rng)

                # Calcular matriz de errores (MSE, MAE, ECCP) para cada candidato
                errores_candidatos = _evaluar_candidatos_paralelo(candidatos, n_cores, semillas_epocas[epoc], pool)

                # Calcular errores ponderados (solo si aplica) para cada candidato
                # Columna 0: MSE, Columna 1: MAE, Columna 2: ECCP
                distancias_candidatos = errores_candidatos[:,0]

                # Condiciones para aceptar candidatos
                condiciones_finitas = np.all(np.isfinite(errores_candidatos), axis=1)
                mascara_aceptacion = condiciones_finitas & (distancias_candidatos <= epsilon_epoch)
                # Actualizar vectores con particulas, errores, distancias
                candidatos_bloques.append(candidatos[mascara_aceptacion])
                errores_bloques.append(errores_candidatos[mascara_aceptacion])
                distancias_bloques.append(distancias_candidatos[mascara_aceptacion])

                # Actualizar contadores
                total_aceptados += np.sum(mascara_aceptacion)
                total_iteraciones += n_simul

                if total_candidatos >= n_max_candidatos:
                    break

            candidatos_aceptados = np.concatenate(candidatos_bloques, axis=0)
            errores_aceptados = np.concatenate(errores_bloques)
            distancias_aceptadas = np.concatenate(distancias_bloques)
            pesos_aceptados = calcular_pesos_aceptados(candidatos_aceptados, poblacion, pesos, mu_0, cov_matriz_0, cov_kernel)

            # Estadísticas
            ess = 1 / np.sum(pesos_aceptados ** 2)

            print(
                f"Epoca {epoc}: iteraciones={total_iteraciones}, aceptados={total_aceptados}"
                f"epsilon_epoch={epsilon_epoch:.4f}"
                f"ESS={ess:.1f}"
            )

            # Actualizar elementos para siguiente epoch
            epsilon_epoch = np.quantile(distancias_aceptadas, 0.50)
            poblacion = candidatos_aceptados
            pesos = pesos_aceptados

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Crear matriz de respuesta
    filas = []
    nombres_gamma = GLM0.params.index

    for theta, errores, distancia, peso in zip(poblacion, errores_aceptados, distancias_aceptadas, pesos):
        gamma_params, delta, beta3, rho = transformar_theta_v1(theta, ngammas, rho_upper_range)

        fila = {
            f"gamma_{nombre}": valor
            for nombre, valor in zip(nombres_gamma, gamma_params)
        }
        fila.update({
            "delta": delta,
            "beta3": beta3,
            "rho": rho,
            "error_mse": errores[0],
            "error_mae": errores[1],
            "error_eccp": errores[2],
            "error": distancia,
            "peso": peso,
        })
    
        filas.append(fila)

    return pd.DataFrame(filas)

def calcular_cov_kernel(poblacion, pesos, nparametros, epoc):
    if epoc == 0:
        cov_kernel = None
    else:
        # Calcular la matriz de covarianza del kernel a partir de la población y los pesos
        # Se emplea un 2 para alcanzar una dispersión mayor en la generación de candidatos
        cov_kernel = 2 * np.cov(poblacion, rowvar=False, aweights=pesos)
        # Se agrega un pequeño valor a la diagonal para evitar problemas de singularidad
        cov_kernel += np.eye(nparametros) * 1e-8
    return cov_kernel

def generar_candidatos(n_simul, poblacion, cov_kernel, pesos, nparametros, epoc, rng):
    if epoc == 0:
        candidatos = poblacion
    else:
        indices = rng.choice(len(poblacion), size=n_simul, p=pesos)
        perturbaciones = rng.multivariate_normal(mean=np.zeros(nparametros), cov=cov_kernel, size=n_simul)
        candidatos = poblacion[indices] + perturbaciones
    return candidatos

def calcular_pesos_aceptados(particulas, poblacion, pesos, mu_0, cov_matriz_0, cov_kernel):
    # Convertir los pesos previos a logaritmos para evitar problemas de subdesbordamiento
    log_pesos_previos = np.log(pesos)
    log_pesos_aceptados = []

    for theta in particulas:
        # Densidad de la distribución del kernel en el denominador
        log_densidades_kernel = multivariate_normal.logpdf(poblacion, mean=theta, cov=cov_kernel)
        log_denominador = logsumexp(log_pesos_previos + log_densidades_kernel)
        # Densidad de la distribución previa en el numerador
        log_numerador = multivariate_normal.logpdf(theta, mean=mu_0, cov=cov_matriz_0)

        log_peso = log_numerador - log_denominador
        log_pesos_aceptados.append(log_peso)

    log_pesos_aceptados = np.asarray(log_pesos_aceptados)
    # Normalizar los pesos aceptados para que sumen 1
    pesos_aceptados = np.exp(log_pesos_aceptados - logsumexp(log_pesos_aceptados))
    return pesos_aceptados