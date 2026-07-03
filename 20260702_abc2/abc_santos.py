# Modelo Approximate Bayesian Computation (ABC)
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import freeze_support
from previoABC import flujo_previo_ABC
from funciones import (
    corrida_simulacion
)

# Número de núcleos para paralelización
n_cores = 10
# Número de simulaciones por núcleo
nsimul = 25000

def main():
    data = flujo_previo_ABC(region = "Los Santos", print_comprobacion_dimensiones = False)

    real = data["train_pivot"].values
    m = data["m_training"]
    design_mat = data["design_mat_train"]
    nsites = data["nsites_training"]
    dist_mat = data["dist_mat_training"]

    print(str(nsites) + ' locaciones')
    print(str(m) + ' meses en training')

    seeds = np.arange(1001, 1001 + n_cores)
    print(f"Ejecutando con {n_cores} núcleos...")

    with Pool(processes = n_cores) as pool:
        resultados = pool.starmap(
            corrida_simulacion,
            [
                (real, m, nsites, design_mat, dist_mat, nsimul, seeds[i])
                for i in range(n_cores)
            ]
        )

    filas_totales = [fila for sublist in resultados for fila in sublist]
    return filas_totales

if __name__ == "__main__":
    freeze_support()
    
    inicio = datetime.now() 
    print("Hora inicial:", inicio)
    
    print("Cantidad de variantes de modelo: 3")
    print("Cantidad de variantes de covariables: 1")
    print("Cantidad de combinaciones de modelos y covariables: 3*1")
    print(f"Cantidad de columnas de resultados: 9+ncov")
    print(f"Dimensiones esperadas del dataframe de resultados: {3*1*n_cores*nsimul} filas x {9} +ncov columnas")

    filas_totales = main()
    df = pd.DataFrame(filas_totales)
    df.to_parquet("20260702_santos.parquet")

    fin = datetime.now()
    print("Hora final:", fin)

    duracion = fin - inicio
    horas, resto = divmod(duracion.total_seconds(), 3600)
    minutos, segundos = divmod(resto, 60)

    print(f"Duración total: {int(horas)} horas, {int(minutos)} minutos, {int(segundos)} segundos")
    print("Dimensiones guardadas: ", df.shape)