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
def corrida_simulacion_exp_errores(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p = 0.01, id_simulacion = None, datos_pivot = None):
    # Filtrar data.frame de parámetros para el modelo específico seleccionado y el percentil de error elegido
    dt_parametros = filtrar_df_modelo_especifico(dt_parametros, D_elegido, M_elegido, "errorMSE", p)
    n_conjunto_parametros = dt_parametros.shape[0]

    if id_simulacion is None:
        id_simulacion = np.random.randint(0, n_conjunto_parametros)

    # Seleccionar secuencia de parametros para un id_simulacion específico (o aleatorio si id_simulacion es None)
    dt_parametros = dt_parametros.iloc[[id_simulacion]]
    
    errorMSE = dt_parametros["errorMSE"].iloc[0]
    errorSWD = dt_parametros["errorSWD"].iloc[0]
    phi12, sigma12, phi, sigma, beta3, rho = dt_parametros[["phi12", "sigma12", "phi", "sigma", "beta3", "rho"]].iloc[0]

    print(f"Calibración GLM:: MSE: {errorMSE:.2f}, SWD: {errorSWD:.2f}, phi12: {phi12}, sigma12: {sigma12} phi: {phi}, sigma: {sigma}, beta3: {beta3}, rho: {rho}")

    # Procesos comunes para todos los sitios
    X1_auxiliar_fijo = simular_logAR12(m, phi12, sigma12).reshape(m, 1) # m x 1
    X2_auxiliar_fijo = simular_logAR1(m, phi, sigma).reshape(m, 1) # m x 1 

    # Procesos específicos para cada sitio (forma matricial)
    X1_auxiliar_completo = simular_logAR12(m, phi12, sigma12, nsites) # m x nsites
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
        print(f"Simulación:: MSE: {errorMSE:.2f}, MAE: {errorMAE:.2f}, SWD: {errorSWD:.2f}")

    return X_train_auxiliar

# Métricas de error para la trayectoria esperada de las precipitaciones 
def metricas_error_presimulaciones(datos_prepivot):
    # Datos en escala original
    Y_pred = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='Y_pred')
    chirps = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='chirps')
    
    errorMSE = mse(Y_pred.values, chirps.values)
    errorMAE = mae(Y_pred.values, chirps.values)
    errorSWD = sliced_wasserstein(Y_pred.values, chirps.values)
    print(f"GLM con:: MSE: {errorMSE:.2f}, MAE: {errorMAE:.2f}, SWD: {errorSWD:.2f}")

# datos_pivot: REQUERIDO! Indica trayectoria esperada de las precipitaciones
def corrida_simulacion_precipitaciones(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p = 0.01, id_simulacion = None, datos_prepivot = None):
    # Datos proyectados previo a convertir escala original
    Y_plus_pred = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='Y_plus_pred').values
    simulacion_exp_errores = corrida_simulacion_exp_errores(m, nsites, dist_mat, D_elegido, M_elegido, dt_parametros, p, id_simulacion)
    # Convertir a escala original
    simulaciones_precitaciones = Y_plus_pred * simulacion_exp_errores - 1
    
    return simulaciones_precitaciones

def metricas_error_simulaciones(simulaciones, datos_prepivot):
    # Datos en escala original
    chirps = datos_prepivot.pivot(index='date', columns=['lat','lon'], values='chirps')
    errorMSE = mse(simulaciones, chirps.values)
    errorMAE = mae(simulaciones, chirps.values)
    errorSWD = sliced_wasserstein(simulaciones, chirps.values)
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



