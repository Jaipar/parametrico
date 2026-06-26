# Preparar datos para el modelo Approximate Bayesian Computation (ABC)
import numpy as np
import pandas as pd
from funciones import (
    preparar_datos_region,
    dividir_train_test,
    parametros_ubicaciones,
    preprocesamiento_escalar_datos,
    calibrar_modelo_GLM,
    GLM_predicciones_residuos_train,
    GLM_predicciones_residuos_test,
    comprobar_dimensiones_ubic
)

def flujo_previo_ABC(region, ruta_datos_precipitacion = 'tablas_precipitaciones.parquet', print_comprobacion_dimensiones = False):
    # Cargar datos de precipitaciones CHIRPS
    datos_precipitacion = pd.read_parquet(ruta_datos_precipitacion)

    # Filtrar datos para cada región
    datos_region = preparar_datos_region(datos_precipitacion, region)
    
    # Fecha corte para training
    fecha_corte_2018 = pd.to_datetime('2018-12-31')

    # Determinar parámetros relacionados con las ubicaciones
    ubic, nsites, dist_mat = parametros_ubicaciones(datos_region)

    # Separar en training y testing
    datos_region_train, datos_region_test, m_training = dividir_train_test(datos_region, ubic, nsites, fecha_corte_2018)

    # Determinar parámetros relacionados con las ubicaciones (training y testing)
    ubic_training, nsites_training, dist_mat_training = parametros_ubicaciones(datos_region_train)

    # Constructores para estandarizar información
    poly_region, scaler_region = preprocesamiento_escalar_datos(datos_region_train)

    # Modelo GLM
    model_GLM_region = calibrar_modelo_GLM(datos_region_train, print_summary = True)

    datos_region_train = GLM_predicciones_residuos_train(datos_region_train, model_GLM_region)
    datos_region_test = GLM_predicciones_residuos_test(datos_region_test, poly_region, scaler_region, model_GLM_region)

    ## Datos de entrenamiento (reales)
    datos_region_train = datos_region_train.sort_values(['lat','lon','date'],ascending=[False,True,True])
    datos_region_train_pivot = datos_region_train.pivot(index='date', columns=['lat','lon'], values='exp_residuos')

    ## Datos para prueba (reales)
    datos_region_test = datos_region_test.sort_values(['lat','lon','date'],ascending=[False,True,True])

    if print_comprobacion_dimensiones:
        print(comprobar_dimensiones_ubic(ubic_training, datos_region_train_pivot))

    return {
        "ubic": ubic,
        "nsites": nsites,
        "dist_mat": dist_mat,
        "train": datos_region_train,
        "train_pivot": datos_region_train_pivot,
        "test": datos_region_test,
        "GLM": model_GLM_region,
        "ubic_training": ubic_training,
        "nsites_training": nsites_training,
        "dist_mat_training": dist_mat_training,
        "m_training": m_training,
    }