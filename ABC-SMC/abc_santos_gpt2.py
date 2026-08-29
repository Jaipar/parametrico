import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Preparar datos para el modelo ABC-SMC 
import numpy as np
import pandas as pd
from datetime import datetime
from funciones_gpt2 import (
    preparar_datos_region,
    dividir_train_test,
    parametros_ubicaciones,
    preprocesamiento_escalar_datos,
    calibrar_modelo_GLM,
    GLM_predicciones_residuos_train,
    GLM_predicciones_residuos_test,
    comprobar_dimensiones_ubic,
    procesamiento_matriz_diseno,
    proceso_abc_smc
)

def flujo_previo_ABC(region, ruta_datos_precipitacion = 'tablas_precipitaciones.parquet', print_comprobacion_dimensiones = True):
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

    datos_region_train = datos_region_train.sort_values(
        ['lat','lon','date'], ascending=[False,True,True]
    ).copy()
    datos_region_test = datos_region_test.sort_values(
        ['lat','lon','date'], ascending=[False,True,True]
    ).copy()

    # Determinar parámetros relacionados con las ubicaciones (training y testing)
    ubic_training, nsites_training, dist_mat_training = parametros_ubicaciones(datos_region_train)

    # Constructores para estandarizar información
    poly_region, scaler_region = preprocesamiento_escalar_datos(datos_region_train)

    # Modelo GLM
    model_GLM_region = calibrar_modelo_GLM(
        datos_region_train,
        print_summary=True,
        poly=poly_region,
        scaler=scaler_region,
    )

    # Matriz diseño
    train_design = procesamiento_matriz_diseno(datos_region_train, poly_region, scaler_region)
    test_design = procesamiento_matriz_diseno(datos_region_test, poly_region, scaler_region)
    train_design = train_design.reindex(columns=model_GLM_region.params.index)
    test_design = test_design.reindex(columns=model_GLM_region.params.index, fill_value=0)

    datos_region_train = GLM_predicciones_residuos_train(datos_region_train, model_GLM_region)
    datos_region_test = GLM_predicciones_residuos_test(datos_region_test, poly_region, scaler_region, model_GLM_region)

    ## Datos de entrenamiento (reales)
    datos_region_train_pivot = datos_region_train.pivot(index='date', columns=['lat','lon'], values='chirps')

    ## Datos para prueba (reales)
    if print_comprobacion_dimensiones:
        print(comprobar_dimensiones_ubic(ubic_training, datos_region_train_pivot))

    return {
        "ubic": ubic,
        "nsites": nsites,
        "dist_mat": dist_mat,
        "train": datos_region_train,
        "Y_train_pivot": datos_region_train_pivot,
        "train_design": train_design,
        "test": datos_region_test,
        "test_design": test_design,
        "GLM": model_GLM_region,
        "ubic_training": ubic_training,
        "nsites_training": nsites_training,
        "dist_mat_training": dist_mat_training,
        "m_training": m_training,
    }

# Número de épocas
nepochs = 20
# Número de simulaciones por época
nsimul = 6000
# Número de núcleos
n_cores = 6
# Número mínimo de partículas en la última época
n_min_ultima_epoca = 5000
# Máximo de candidatos permitidos para alcanzar ese mínimo
max_candidatos_ultima_epoca = 120000

def main():
    data = flujo_previo_ABC(region = "Los Santos", print_comprobacion_dimensiones = False)

    real = data["Y_train_pivot"].values
    m = data["m_training"]
    nsites = data["nsites_training"]
    dist_mat = data["dist_mat_training"]
    design_mat = data["train_design"].to_numpy()
    GLM0 = data["GLM"]
    meses = data["Y_train_pivot"].index.month.to_numpy()

    print(str(nsites) + ' locaciones')
    print(str(m) + ' meses en training')

    seed = 1000
    print(f"Ejecutando con {nepochs} épocas, {nsimul} simulaciones por época y {n_cores} núcleos")

    filas_totales = proceso_abc_smc(
        real,
        m,
        nsites,
        dist_mat,
        design_mat,
        GLM0,
        nepochs,
        nsimul,
        seed,
        n_cores=n_cores,
        meses=meses,
        cuantil_umbral=0.90,
        n_min_output=n_min_ultima_epoca,
        max_candidatos_ultima_epoca=max_candidatos_ultima_epoca,
    )
    return filas_totales

if __name__ == "__main__":
    inicio = datetime.now() 
    print("Hora inicial:", inicio)
    
    print("Cantidad de variantes de modelo: 1")
    print("Cantidad de variantes de covariables: 1")

    filas_totales = main()
    filas_totales.to_parquet("20260823_santosv2.parquet")

    fin = datetime.now()
    print("Hora final:", fin)

    duracion = fin - inicio
    horas, resto = divmod(duracion.total_seconds(), 3600)
    minutos, segundos = divmod(resto, 60)

    print(f"Duración total: {int(horas)} horas, {int(minutos)} minutos, {int(segundos)} segundos")
    print("Dimensiones guardadas: ", filas_totales.shape)
