import pandas as pd
import matplotlib.pyplot as plt
import ast  # para convertir strings que parecen listas en listas reales
import pandas as pd
import os

aa = pd.read_parquet('resultados_abc_los_santos1.parquet')
bb = pd.read_parquet('resultados_abc_los_santos2.parquet')
cc = pd.read_parquet('resultados_abc_los_santos3.parquet')

aa=pd.concat([aa,bb,cc])
df = aa[(aa.modelo =='D4') & (aa.covariables =='M1')]
df = df.sort_values('error',ascending=True).head(5000)
print(df.info())
# Convertimos strings a listas si es necesario


# Carpeta donde guardaremos las imágenes
carpeta = "histogramas_aplicacion_abc"
os.makedirs(carpeta, exist_ok=True)

# Columnas numéricas
columnas_numericas = ['phi','sigma','beta3', 'rho']

# 1️⃣ Histograma de columnas numéricas
fig, axs = plt.subplots(len(columnas_numericas), 1, figsize=(10, 5 * len(columnas_numericas)))
for i, col in enumerate(columnas_numericas):
    axs[i].hist(df[col], bins=30, color='skyblue', edgecolor='black')
    axs[i].set_title(f'Histograma de {col}')
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Frecuencia')
    axs[i].grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.savefig(os.path.join(carpeta, "histogramas_numericos.png"))
plt.close()

print(f"Se crearon 3 imágenes en la carpeta '{carpeta}':")
print("- histogramas_numericos_D4_M1.png")
