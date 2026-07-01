import numpy as np
from funciones import *

def filtrar_df_por_percentil(df, nombre_columna, p = 0.05):
    umbral = df[nombre_columna].quantile(p)
    df_filtrado = df[df[nombre_columna] <= umbral]
    return df_filtrado

def filtrar_df_modelo_especifico(df, D_elegido, M_elegido, columna_error, p = 0.05):
    df_filtrado = df.loc[(df["modelo"] == D_elegido) & (df["covariables"] == M_elegido)]
    df_filtrado = filtrar_df_por_percentil(df_filtrado, columna_error, p)
    return df_filtrado

def listado_percentiles(df, percentiles = [0.002, 0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.10, 0.25]):
    cuantiles = df[["errorSWD", "errorMSE"]].quantile(percentiles)
    resultados = cuantiles.reset_index().rename(columns={
        "index": "percentil",
        "errorSWD": "umbral_SWD",
        "errorMSE": "umbral_MSE"
    })
    return resultados

def resumen_errores(df, sort_by = 'mean_errorSWD'):
    resumen = df.groupby(['modelo', 'covariables'], as_index=False).agg(
        mean_errorSWD = ('errorSWD', 'mean'), 
        sd_errorSWD = ('errorSWD', 'std'),
        mean_errorMSE = ('errorMSE', 'mean'),
        sd_errorMSE = ('errorMSE', 'std'),
        min_errorSWD = ('errorSWD', 'min'),
        max_errorSWD = ('errorSWD', 'max'),
        min_errorMSE = ('errorMSE', 'min'),
        max_errorMSE = ('errorMSE', 'max'),
        iqr_SWD = ('errorSWD', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        iqr_MSE = ('errorMSE', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        p25_SWD = ('errorSWD', lambda x: np.percentile(x, 25)),
        p25_MSE = ('errorMSE', lambda x: np.percentile(x, 25)),
        p75_SWD = ('errorSWD', lambda x: np.percentile(x, 75)),
        p75_MSE = ('errorMSE', lambda x: np.percentile(x, 75)),
        n = ('modelo', 'size')
    ).sort_values(sort_by, ascending=True)
    return resumen

def seleccionar_mejor_modelo(df, sort_by = 'mean_errorSWD', posicion_n_ranking = 0):
    resumen = resumen_errores(df, sort_by)
    fila = resumen.iloc[posicion_n_ranking]
    return fila["modelo"], fila["covariables"] # (Modelo D, Covariables M)

def umbrales_por_modelo(df, columna_error, p = 0.05):
    umbral = df.groupby(['modelo', 'covariables'])[columna_error].quantile(p).reset_index()
    umbral = umbral.rename(columns={columna_error: 'umbral'})
    return umbral

def filtrar_df_errores_por_percentil_agrupado(df, columna_error, p = 0.05):
    umbral = umbrales_por_modelo(df, columna_error, p)
    df_filtrado = df.merge(umbral, on=['modelo', 'covariables'])
    df_filtrado = df_filtrado[df_filtrado[columna_error] <= df_filtrado['umbral']]
    return df_filtrado

# m: número de (tiempos) de datos de precipitación
# nsites: número de sitios (ubicaciones) para simulación
# dist_mat: matriz de distancias entre ubicaciones
# D_elegido: identificador del tipo de modelo elegido
# M_elegido: identificador del conjunto de covariables elegidas
# dt_parametros: data.frame con parametros calibrados (mediante ABC)
#   para simulación de procesos X_1t(s), X_2t(s) y X_3t(s)
# p: percentil para filtrar parámetros según error (default 0.01 de errorMSE)
# id_simulacion: índice para seleccionar un conjunto específico de parámetros (si None, se genera al azar)
# datos_pivot: se procede a calcular medidas de error en caso de existir datos reales para comparar

# Note que alpha_(s,t) es determínistico y X_1t(s), X_2t(s) y X_3t(s) varían por simulación
def corrida_simulacion_exp_errores(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p = 0.01, id_simulacion = None, datos_pivot = None, boolean_imprimir = True):
    # Filtrar data.frame de parámetros para el modelo específico seleccionado y el percentil de error elegido
    dt_parametros = filtrar_df_modelo_especifico(dt_parametros, D_elegido, M_elegido, "errorMSE", p)
    n_conjunto_parametros = dt_parametros.shape[0]

    if id_simulacion is None:
        id_simulacion = np.random.randint(0, n_conjunto_parametros)

    # Seleccionar secuencia de parametros para un id_simulacion específico (o aleatorio si id_simulacion es None)
    dt_parametros = dt_parametros.iloc[[id_simulacion]]
    
    errorMSE = dt_parametros["errorMSE"].iloc[0]
    errorSWD = dt_parametros["errorSWD"].iloc[0]
    delta, phi, sigma, beta3, rho = dt_parametros[["delta", "phi", "sigma", "beta3", "rho"]].iloc[0]

    if boolean_imprimir:
        print(f"Calibración con errorMSE:: {errorMSE:.2f}, errorSWD:: {errorSWD:.2f}, delta: {delta}, phi: {phi}, sigma: {sigma}, beta3: {beta3}, rho: {rho}")

    # Procesos comunes para todos los sitios
    X1_auxiliar_fijo = simular_X1(m, delta).reshape(m, 1) # m x 1
    X2_auxiliar_fijo = simular_logAR1(m, phi, sigma).reshape(m, 1) # m x 1 

    # Procesos específicos para cada sitio (forma matricial)
    X1_auxiliar_completo = simular_X1(m, delta, nsites) # m x nsites
    X2_auxiliar_completo = simular_logAR1(m, phi, sigma, nsites) # m x nsites
    X3_auxiliar_completo = simular_X3(m, rho, beta3, nsites, dist_mat) # m x nsites

    # Calcular precipitaciones de acuerdo a diferentes D desde D3 a D8
    if D_elegido == "D3":
        X_train_auxiliar = X1_auxiliar_fijo * X3_auxiliar_completo
    elif D_elegido == "D4":
        X_train_auxiliar = X1_auxiliar_completo * X3_auxiliar_completo
    elif D_elegido == "D5":
        X_train_auxiliar = X1_auxiliar_fijo * X2_auxiliar_fijo * X3_auxiliar_completo
    elif D_elegido == "D6":
        X_train_auxiliar = X1_auxiliar_completo * X2_auxiliar_fijo * X3_auxiliar_completo
    elif D_elegido == "D7":
        X_train_auxiliar = X1_auxiliar_fijo * X2_auxiliar_completo * X3_auxiliar_completo
    elif D_elegido == "D8":
        X_train_auxiliar = X1_auxiliar_completo * X2_auxiliar_completo * X3_auxiliar_completo
    
    if datos_pivot is not None:
        real = datos_pivot.values
        errorMSE = mse(X_train_auxiliar, real)
        errorMAE = mae(X_train_auxiliar, real)
        errorSWD = sliced_wasserstein(X_train_auxiliar, real)
        if boolean_imprimir:
            print(f"Simulación con:: MSE: {errorMSE:.2f}, MAE: {errorMAE:.2f}, SWD: {errorSWD:.2f}")

    return X_train_auxiliar

# datos_pivot: REQUERIDO! Indica trayectoria esperada de las precipitaciones
def corrida_simulacion_precipitaciones(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p = 0.01, id_simulacion = None, datos_prepivot = None, boolean_imprimir = True):
    logY_pred = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='logY_pred')
    exp_simulacion_errores = corrida_simulacion_exp_errores(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p, id_simulacion, boolean_imprimir = boolean_imprimir)
    simulaciones_precitaciones = np.exp(logY_pred.values) * exp_simulacion_errores - 1
    return simulaciones_precitaciones

def return_metricas_error(estimaciones, chirps, calcular_swd=True):
    return mse(estimaciones, chirps), mae(estimaciones, chirps), sliced_wasserstein(estimaciones, chirps) if calcular_swd else np.nan

# Métricas de error para la trayectoria esperada de las precipitaciones 
def print_metricas_error_presimulaciones(datos_prepivot):
    # Datos en escala original
    Y_pred = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='Y_pred').values
    chirps = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='chirps').values
    errorMSE, errorMAE, errorSWD = return_metricas_error(Y_pred, chirps)
    print(f"Regresión lineal con:: MSE: {errorMSE:.2f}, MAE: {errorMAE:.2f}, SWD: {errorSWD:.2f}")

def print_metricas_error_simulaciones(simulaciones, datos_prepivot):
    # Datos en escala original
    chirps = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='chirps').values
    errorMSE, errorMAE, errorSWD = return_metricas_error(simulaciones, chirps)
    print(f"Simulación con:: MSE: {errorMSE:.2f}, MAE: {errorMAE:.2f}, SWD: {errorSWD:.2f}")

def asignar_coords_a_simulacion(simulaciones, datos_pivot, fechas = None):
    t, d = simulaciones.shape
    if fechas is None:
        # Extraer fechas directamente de datos_pivot
        fechas = datos_pivot.index
    lats = datos_pivot.columns.get_level_values(0).values
    lons = datos_pivot.columns.get_level_values(1).values

    df = pd.DataFrame({'date': np.repeat(fechas, d),
        'lat': np.tile(lats, t),
        'lon': np.tile(lons, t),
        'simulaciones': simulaciones.flatten()
    })

    return df

def corrida_n_simulaciones_precipitaciones(m, nsites, dist_mat, D_elegido, M_elegido, 
                                           dt_parametros,  n_simulaciones = 100, semilla_inicial = 1000, p = 0.01,
                                           datos_prepivot = None, boolean_imprimir = False):
    # Filtrar data.frame de parámetros para el modelo específico seleccionado y el percentil de error elegido
    n_conjunto_parametros = filtrar_df_modelo_especifico(dt_parametros, D_elegido, M_elegido, "errorMSE", p).shape[0]
    
    # Semillas para obtener id_filas en generación de simulaciones
    semillas = np.arange(semilla_inicial, semilla_inicial + n_simulaciones)
    
    simulaciones = np.zeros((n_simulaciones, m, nsites))
    for i in range(n_simulaciones):
        seed = semillas[i]
        rng = np.random.default_rng(seed)

        id_parametro = rng.integers(0, n_conjunto_parametros)
        simulaciones[i, :, :] = corrida_simulacion_precipitaciones(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p, id_parametro, datos_prepivot, boolean_imprimir)

    return simulaciones

def comparar_test_DyM(test_prepivot, m, nsites, dist_mat, dt_parametros, n_simulaciones=1000, p=0.01, calcular_swd=False):
    chirps = test_prepivot.pivot(index='date', columns=['lat','lon'], values='chirps').values
    filas_metricas = []

    modelosDyM = dt_parametros[["modelo","covariables"]].drop_duplicates().sort_values(["modelo","covariables"])

    for _, fila in modelosDyM.iterrows():
        D = fila["modelo"]
        M = fila["covariables"]

        simulaciones = corrida_n_simulaciones_precipitaciones(
            m, nsites, dist_mat, D, M,
            dt_parametros, n_simulaciones,
            p=p, datos_prepivot=test_prepivot
        )
    
        for i in range(n_simulaciones):
            errorMSE, errorMAE, errorSWD = return_metricas_error(simulaciones[i, :, :], chirps, calcular_swd)
            filas_metricas.append({
            "D": D,
            "M": M,
            "errorMSE": errorMSE,
            "errorMAE": errorMAE,
            "errorSWD": errorSWD
            })
            
    metricas = pd.DataFrame(filas_metricas)
    resumen = (
        metricas
        .groupby(["D", "M"])
        .agg(
            MSE_mean=("errorMSE", "mean"),
            MSE_std=("errorMSE", "std"),
            MAE_mean=("errorMAE", "mean"),
            MAE_std=("errorMAE", "std"),
            SWD_mean=("errorSWD", "mean"),
            SWD_std=("errorSWD", "std"),
        )
        .reset_index()
        .sort_values("MSE_mean")
    )

    return resumen