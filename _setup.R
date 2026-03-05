# Carga e instala (si no están instalados) los paquetes necesarios
if (!require("pacman")) install.packages("pacman")

# Manejo y análisis de datos geoespaciales (vectores)
pacman::p_load(sf)

# Visualización temática de datos espaciales (mapas estáticos y dinámicos)
pacman::p_load(tmap)
tmap_mode("plot")

# Manipulación y transformación de datos (data frames y tibbles)
pacman::p_load(dplyr, tidyr)

# Escritura y lectura de datos en formato .feather o .parquet
pacman::p_load(arrow)

# Visualización interactiva de mapas web (basados en Leaflet.js)
pacman::p_load(leaflet)

# Creación de gráficos estáticos y personalizados (gramática de gráficos)
pacman::p_load(ggplot2)

# Enlace con servidores CHIRPS
pacman::p_load(chirps)

# Consultar elevación puntual
pacman::p_load(elevatr)

# Tratamiento de fechas
pacman::p_load(lubridate)

# Paletas de colores para gráficos
pacman::p_load(RColorBrewer)