# Limitaciones Argumentos usados en paralelizacion
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Preparar datos para el modelo ABC-SMC 
import pandas as pd
from datetime import datetime
from funciones_final import (
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
    datos_train, datos_test, m_training = dividir_train_test(datos_region, ubic, nsites, fecha_corte_2018)
    datos_train = datos_train.sort_values(['lat','lon','date'], ascending=[False,True,True]).copy()
    datos_test = datos_test.sort_values(['lat','lon','date'], ascending=[False,True,True]).copy()

    # Determinar parámetros relacionados con las ubicaciones (training)
    ubic_training, nsites_training, dist_mat_training = parametros_ubicaciones(datos_train)

    # Constructores para estandarizar información
    poly, scaler = preprocesamiento_escalar_datos(datos_train)

    # Modelo GLM
    GLM = calibrar_modelo_GLM(datos_train, print_summary=True, poly=poly, scaler=scaler)

    # Matriz diseño
    train_design = procesamiento_matriz_diseno(datos_train, poly, scaler).reindex(columns=GLM.params.index)
    test_design = procesamiento_matriz_diseno(datos_test, poly, scaler).reindex(columns=GLM.params.index)

    datos_train = GLM_predicciones_residuos_train(datos_train, GLM)
    datos_test = GLM_predicciones_residuos_test(datos_test, poly, scaler, GLM)

    ## Datos de entrenamiento (pivot)
    datos_train_pivot = datos_train.pivot(index='date', columns=['lat','lon'], values='chirps')

    ## Comrprobar el orden de las ubicaciones es correcto
    if print_comprobacion_dimensiones:
        print(comprobar_dimensiones_ubic(ubic_training, datos_train))

    return {
        "ubic": ubic,
        "nsites": nsites,
        "dist_mat": dist_mat,
        "train": datos_train,
        "Y_train_pivot": datos_train_pivot,
        "train_design": train_design,
        "test": datos_test,
        "test_design": test_design,
        "GLM": GLM,
        "ubic_training": ubic_training,
        "nsites_training": nsites_training,
        "dist_mat_training": dist_mat_training,
        "m_training": m_training,
    }

# Número de épocas
n_epochs = 20
# Número de simulaciones por época
n_simul = 6000
# Número de núcleos
n_cores = 6
# Número mínimo de partículas aceptadas por época
n_min_aceptados = int(n_simul/2)

def main():
    data = flujo_previo_ABC(region = "Los Santos", print_comprobacion_dimensiones = False)

    real = data["Y_train_pivot"].values
    m = data["m_training"]
    nsites = data["nsites_training"]
    dist_mat = data["dist_mat_training"]
    design_mat = data["train_design"].to_numpy()
    GLM0 = data["GLM"]

    print(str(nsites) + ' locaciones')
    print(str(m) + ' meses en training')

    seed = 1000
    print(
        f"Ejecutar {n_epochs} épocas, usando {n_simul} simulaciones por iteración "
        f"hasta alcanzar {n_min_aceptados} candidatos aceptados por época, "
        f"usando {n_cores} núcleos."
    )

    filas_totales = proceso_abc_smc(
        real,
        m,
        nsites,
        dist_mat,
        design_mat,
        GLM0,
        n_cores,
        n_epochs,
        n_simul,
        n_min_aceptados,
        seed
    )
    return filas_totales

if __name__ == "__main__":
    inicio = datetime.now() 
    print("Hora inicial:", inicio)

    filas_totales = main()
    filas_totales.to_parquet("20260830_santos.parquet")

    fin = datetime.now()
    print("Hora final:", fin)

    duracion = fin - inicio
    horas, resto = divmod(duracion.total_seconds(), 3600)
    minutos, segundos = divmod(resto, 60)

    print(f"Duración total: {int(horas)} horas, {int(minutos)} minutos, {int(segundos)} segundos")
    print("Dimensiones guardadas: ", filas_totales.shape)
