# SAB_tipo1
# 
#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

from cProfile import label
import warnings
import re
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import geopandas as gpd
import pickle
import shap
import time as t
import seaborn as sns
import gc
import ast
import math

import numpy as np
import shapely
from shapely.geometry import box, Polygon

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm

import itertools
from itertools import product

import geopy.distance
from geopy.distance import geodesic

from shapely.wkt import loads
from shapely.geometry import Polygon, Point, LineString, box, MultiLineString 
from shapely.ops import nearest_points
from shapely.ops import unary_union
from shapely.ops import split
from shapely.affinity import scale as shapely_scale, translate

from datetime import datetime, time
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from turtle import color

start_time = t.time()

print("Antes de ejecutar el código asegura que tengas los datos del día que quieres analizar.")
### SELECCIONAR LA FECHA
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
mes_a_numero = {mes: numero for numero, mes in enumerate(meses, start=1)}

entrada = input("el mes seleccionado es (enero, febrero, etc.): ")
mes_input = entrada.strip().capitalize()
mes_sel = f"{mes_a_numero[mes_input]:02d}"

### DIRECTORIOS - ACC Madrid Norte
PATH_SECTOR_DATA = 'F:\\Users\\Lai\\Datos\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'
PATH_flujos = 'F:\\Users\\Lai\\original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\'
PATH_bordes = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\'
PATH_mallado = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

carpeta_mes = f"{mes_input}2022"

if mes_input == "Junio":
        PATH_TRAFICO_CELDA = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\mallado_20x20\\'
        PATH_TRAFICO = f'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\{carpeta_mes}\\'
        PATH_COMPLEJIDAD = f'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\{carpeta_mes}\\test\\'
else:
        # Rutas estándar para el resto de los meses (Enero, Abril, Agosto, etc.)
        PATH_TRAFICO_CELDA = f'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\{carpeta_mes}\\'
        PATH_TRAFICO = f'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\test\\{carpeta_mes}\\'
        PATH_COMPLEJIDAD = f'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\{carpeta_mes}\\'

PATH_resultados = f'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\{carpeta_mes}\\test\\'
PATH_sabs = f'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\{carpeta_mes}\\'

# configuración del estudio (ejemplo: CNF5A, CNF5B, etc.)
#! PENDIENTE DE AUTOMATIZAR LA SELECCIÓN DE CONFIGURACIÓN (ej. con input o argumentos)
configuracion_estudio = 'CNF5A'

# Día seleccionado
dia_sel = input(f"Introduce un día del {mes_input} de 2022 (2022-{mes_sel}-DD): ")
dia_sel = f"{int(dia_sel):02d}"  # Asegura que el día tenga dos dígitos

fecha_sel = f"2022-{mes_sel}-{dia_sel}"  # Formatear con ceros a la izquierda (ej. 01, 02, ..., 30) 
### franja de estudio
dia_horas = input("Introduce si quiere estudair un día entero o una franja de horas (dia/hora): ").strip().lower()

if dia_horas == 'dia':
    hora_ini = time(0, 0, 0) 
    hora_fin = time(23, 59, 59) 
    hora_ini_str = '00'
    hora_fin_str = '24'    
elif dia_horas == 'hora':
    hora_ini_str = input("Introduce la hora de inicio (ej. 08, 12): ")
    hora_ini_str = f"{int(hora_ini_str):02d}"  # Asegura que la hora tenga dos dígitos
    hora_fin_str = input("Introduce la hora de fin (ej. 08, 12): ")
    hora_fin_str = f"{int(hora_fin_str):02d}"
    hora_ini = time(int(hora_ini_str), 0, 0) 
    hora_fin = time(int(hora_fin_str), 0, 0) 
    
#--------------------------------------------------
nombre_carpeta = f"RESULTADOS_{fecha_sel}"
# Unir la ruta base con la nueva carpeta
PATH_COMPLEJIDAD_DIA = os.path.join(PATH_COMPLEJIDAD, nombre_carpeta)

# Tamaño de celda en nm
cell_size_nm = 20

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

### Datos de flujos, celdas, sabs obtenidos, sectores y de mallado------------------------------------------------------------------------------- 

# DATASET ANÁLISIS FLUJOS POR CELDA: qué flujos atraviesan cada celda
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo.pkl')

# Datos de los sabs detectados
# SAB tipo 2
sab_tipo2 = pd.read_pickle(PATH_sabs + f'SAB_tipo2_{fecha_sel}.pkl')

# SAB tipo 1
if dia_horas == 'hora':
    ruta_sab1 = PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}_{hora_ini_str}-{hora_fin_str}.pkl'
elif dia_horas == 'dia':
    ruta_sab1 = PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}.pkl'
else:
    ruta_sab1 = ""

if os.path.exists(ruta_sab1):
    sab_tipo1 = pd.read_pickle(ruta_sab1)
    print(f"Archivo de SAB tipo 1 cargado desde {ruta_sab1}.")
else:
    sab_tipo1 = pd.DataFrame()  # Crea un DataFrame vacío
    print(f"Aviso: No se detectó el archivo en {ruta_sab1}. Se asignó sab_tipo1 vacío.")

# Datos del mallado de tamaño 20x20 nm
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')

# Datos de los sectores (con su geometría)
DF_info_conf = pd.read_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')

# Datos de los flujos clusterizados
df_espinazos = pd.read_pickle(PATH_sabs + f'espinazos_flujos_{fecha_sel}.pkl')

## Procesado para obtener Geodataframes------------------------------------------------------------------------------- 
gdf_cells = gpd.GeoDataFrame(DF_cells, geometry='Polygon')
gdf_sab2 = gpd.GeoDataFrame(sab_tipo2, geometry='forma de SAB')

gdf_mallado = gpd.GeoDataFrame(df_mallado, geometry='Polygon')
gdf_mallado = gdf_mallado.rename_geometry('geometry')

### Procesado para obtener las fronteras internas de los sectores------------------------------------------------------------------------------- 

# 1. Aseguramos que los sectores están en un GeoDataFrame
gdf_sectores = gpd.GeoDataFrame(
    DF_info_conf.rename(columns={'Contorno Sector': 'geometry'}),
    geometry='geometry',
    crs=gdf_mallado.crs
)

contorno_externo_global = gdf_sectores.union_all().boundary

fronteras_internas = []

# 2. Iterar por cada sector para limpiar su frontera
for _, sector in gdf_sectores.iterrows():
    sector_id = sector['SECTOR_ID']
    geometria_sector = sector['geometry']

    # Obtenemos el perímetro completo de este sector específico
    perimetro_sector = geometria_sector.boundary
    
    # RESTA ESTRATÉGICA: Al perímetro del sector le quitamos lo que sea contorno global
    # Usamos un buffer minúsculo (1e-7) para absorber errores de precisión numérica
    solo_interno = perimetro_sector.difference(contorno_externo_global.buffer(1e-7))
    
    # 3. Almacenar solo si el resultado contiene líneas (ignoramos puntos o vacíos)
    if not solo_interno.is_empty:
        if solo_interno.geom_type == 'LineString':
            frontera_id = 1
            fronteras_internas.append({
                'ID': frontera_id,
                'SECTOR_ID': sector_id,
                'geometry': solo_interno})
            
        elif solo_interno.geom_type == 'MultiLineString':
            # enumerate(..., start=1) cuenta automáticamente desde el 1
            for i, line in enumerate(solo_interno.geoms, start=1):
                fronteras_internas.append({
                    'ID': i,
                    'SECTOR_ID': sector_id,
                    'geometry': line
                })


# 4. Crear el GeoDataFrame final
print(f"Fronteras internas encontradas: {fronteras_internas}")

# SOLUCIÓN: Pasamos la lista completa y le indicamos que la columna de geometría se llama 'geometry'
gdf_fronteras_internas = gpd.GeoDataFrame(fronteras_internas, geometry='geometry', crs=gdf_sectores.crs)
print(f"Procesados {len(gdf_fronteras_internas)} sectores con fronteras internas.")


### 5. Visualización de las fronteras internas
fig, ax = plt.subplots(figsize=(12, 10))

# Dibujar el área de los sectores como fondo tenue
gdf_sectores.plot(ax=ax, color='whitesmoke', edgecolor='none', alpha=0.4)

# Dibujar el CONTORNO EXTERIOR GLOBAL (lo que hemos eliminado)
# Aparecerá como una línea discontinua roja rodeando todo el espacio
gpd.GeoSeries([contorno_externo_global]).plot(
    ax=ax, color='red', linewidth=1, linestyle='--', label='Límite Externo Eliminado'
)

# Dibujar las FRONTERAS INTERNAS resultantes
# Cada sector tendrá sus líneas internas en un color sólido
gdf_fronteras_internas.plot(
    ax=ax, column='SECTOR_ID', cmap='tab10', linewidth=2.5, label='Fronteras Internas'
)

# Añadir etiquetas de Sector para claridad
for _, row in gdf_sectores.iterrows():
    centro = row['geometry'].centroid
    ax.text(centro.x, centro.y, row['SECTOR_ID'], fontsize=8, ha='center', alpha=0.7)

plt.title("Fronteras Internas por Sector (Excluyendo Perímetro del ACC)", fontsize=14)
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Crear leyenda personalizada
custom_lines = [Line2D([0], [0], color='red', lw=1, linestyle='--'),
                Line2D([0], [0], color='blue', lw=2.5)]
ax.legend(custom_lines, ['Límite Externo (Borrado)', 'Fronteras Internas'], loc='upper right')

plt.show()

### Mapa de los contornos de sectores y el mallado en fondo
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 10))

# 1. Dibujar el mallado de las celdas en el fondo
# Usamos facecolor='none' para que las celdas estén huecas y alpha=0.3 para hacer las líneas transparentes
gdf_mallado.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.3, linewidth=0.5)

# Si prefieres que las celdas tengan un color de fondo semitransparente en lugar de solo los bordes,
# comenta la línea anterior y descomenta la siguiente:
# gdf_mallado.plot(ax=ax, color='lightsteelblue', edgecolor='gray', alpha=0.2, linewidth=0.5)

# 2. Dibujar SOLO los contornos de los sectores
# Usamos .boundary para extraer solo las líneas del perímetro, sin relleno
gdf_sectores.boundary.plot(ax=ax, color='black', linewidth=2)

gdf_sectores.plot(
    ax=ax, 
    column='SECTOR_ID', 
    cmap='Pastel1', 
    alpha=0.5,           # Ajusta este valor (0.1 a 1) para la intensidad del color
    edgecolor='none'
)

# 3. Añadir etiquetas de Sector para mayor claridad (opcional)
for _, row in gdf_sectores.iterrows():
    centro = row['geometry'].centroid
    ax.text(centro.x, centro.y, row['SECTOR_ID'], fontsize=9, ha='center', fontweight='bold', color='black')

# 4. Detalles del gráfico
plt.title("ACC Madrid Norte - Mallado de Celdas", fontsize=14)
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Leyenda personalizada
custom_lines = [Line2D([0], [0], color='gray', lw=0.5, alpha=0.5),
                Line2D([0], [0], color='black', lw=2)]
ax.legend(custom_lines, ['Mallado de Celdas', 'Contorno de Sectores'], loc='upper right')

# Mostrar la figura
plt.show()

### mapa del mallado de cada sector
fig, ax = plt.subplots(figsize=(12, 10))

# 3. Dibujar el mallado de celdas
# Usamos 'column' para dar un color distinto a cada sector
# Usamos 'edgecolor' y 'linewidth' para mantener visible la forma de cada celda
gdf_mallado.plot(
    ax=ax,
    column='Sector',       # Columna del GeoDataFrame que diferencia los sectores
    cmap='Pastel1',        # Mapa de colores (puedes usar 'Set1', 'Set3', etc.)
    alpha=0.7,             # Intensidad del color de relleno (0.1 a 1)
    edgecolor='gray',      # Borde de las celdas para que se vea el mallado
    linewidth=0.5,         # Grosor de las líneas del mallado
    legend=True,           # Habilitar la leyenda automáticamente
    legend_kwds={'title': "Sectores", 'loc': 'upper right'} # Posición de la leyenda
)

# 4. Detalles del gráfico
plt.title("ACC Madrid Norte - Mallado de Celdas por Sector", fontsize=14)
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Mostrar la figura
plt.show()



# ### INICIO DEL BLOQUE DE FILTRADO PARA LA FRANJA HORARIA ESTABLECIDA-----------------
def filtrado_datos(PATH_resultados, PATH_TRAFICO_CELDA, fecha_sel, dia_sel, 
                           dia_horas, hora_ini=None, hora_fin=None, 
                           hora_ini_str="", hora_fin_str=""):
    """
    Comprueba si existen los datasets filtrados. Si existen, los carga. 
    Si no, carga los datos brutos, filtra según la franja horaria/día, 
    los guarda en disco y los devuelve.
    """
    
    # 1. Definir las rutas de los archivos esperados
    if dia_horas == 'hora':
        file_flujos_pkl = f'{PATH_resultados}dataset_celdas_por_flujo_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.pkl'
        file_cells_pkl = f'{PATH_resultados}dataset_flujos_por_celda_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.pkl'
    else:
        file_flujos_pkl = f'{PATH_resultados}dataset_celdas_por_flujo_{dia_sel}_filtrado.pkl'
        file_cells_pkl = f'{PATH_resultados}dataset_flujos_por_celda_{dia_sel}_filtrado.pkl'

    # 2. Comprobar si los archivos ya existen en el disco
    if os.path.exists(file_flujos_pkl) and os.path.exists(file_cells_pkl):
        print("Archivos filtrados detectados. Cargando datos desde disco...")
        df_flujos = pd.read_pickle(file_flujos_pkl)
        df_cells = pd.read_pickle(file_cells_pkl)
        return df_flujos, df_cells

    # 3. Si no existen, iniciamos el proceso de filtrado
    print("Archivos filtrados no encontrados. Iniciando filtrado de datos...")
    
    # Cargar datasets en bruto (raw)
    df_data_cell = pd.read_pickle(PATH_TRAFICO_CELDA + f'DF_T_REAL_CELDA_{fecha_sel}.pkl')
    df_celdas_por_flujo_raw = pd.read_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')
    df_flujos_por_celda_raw = pd.read_pickle(PATH_resultados + 'dataset_flujos_por_celda.pkl')

    df_referencia = df_data_cell.copy()

    # Aplicar el filtro de tiempo si está configurado por horas
    if dia_horas == 'hora' and hora_ini is not None and hora_fin is not None:
        df_referencia["IOBT"] = pd.to_datetime(df_referencia["IOBT"])
        df_referencia = df_referencia[df_referencia["IOBT"].dt.time.between(hora_ini, hora_fin)]

    # Extraer los flujos únicos del día/hora clave
    flujos_dia_clave = set(df_referencia['Clave_Flujo'].unique())

    # Filtrar 'dataset_celdas_por_flujo'
    df_flujos = df_celdas_por_flujo_raw[
        df_celdas_por_flujo_raw['Clave_Flujo'].isin(flujos_dia_clave)
    ].copy()

    # Filtrar celdas que tengan flujos tras el filtro
    df_cells = df_flujos_por_celda_raw[
        df_flujos_por_celda_raw['Flujos_Clusterizados'].map(len) > 0
    ].copy()

    print(f"Filtrado listo: {len(df_flujos)} flujos y {len(df_cells)} celdas activas.")

    # 4. Guardar los resultados generados para futuras ejecuciones
    file_flujos_csv = file_flujos_pkl.replace('.pkl', '.csv')
    file_cells_csv = file_cells_pkl.replace('.pkl', '.csv')

    df_flujos.to_csv(file_flujos_csv, index=False)
    df_cells.to_csv(file_cells_csv, index=False)
    df_flujos.to_pickle(file_flujos_pkl)
    df_cells.to_pickle(file_cells_pkl)

    return df_flujos, df_cells

df_flujos, df_cells = filtrado_datos(
    PATH_resultados = PATH_resultados,
    PATH_TRAFICO_CELDA = PATH_TRAFICO_CELDA,
    fecha_sel = fecha_sel,
    dia_sel = dia_sel,
    dia_horas = dia_horas,
    hora_ini = hora_ini,        # Solo necesario si dia_horas == 'hora'
    hora_fin = hora_fin,        # Solo necesario si dia_horas == 'hora'
    hora_ini_str = hora_ini_str,
    hora_fin_str = hora_fin_str
)

# # 1. Cargar el archivo de reference (vuelos reales del día 01)
# # df_referencia = pd.read_pickle(PATH_TRAFICO_CELDA + f'DF_T_REAL_CELDA_20x20_{fecha_sel}.pkl')
# df_data_cell = pd.read_pickle(PATH_TRAFICO_CELDA + f'DF_T_REAL_CELDA_{fecha_sel}.pkl')

# df_referencia = df_data_cell.copy()
# df_referencia["IOBT"] = pd.to_datetime(df_referencia["IOBT"])

# # Aplicar el filtro ocultando la fecha y quedándote solo con la hora
# df_referencia = df_referencia[df_referencia["IOBT"].dt.time.between(hora_ini, hora_fin)]

# # (Opcional) Verificar el resultado
# print(df_referencia["IOBT"].head())

# # 2. Cargar los datasets que queremos filtrar
# # NOTA: iguales que anteriores pero son apra que no influyen a los anteriores datasets
# df_celdas_por_flujo_raw = pd.read_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')
# df_flujos_por_celda_raw = pd.read_pickle(PATH_resultados + 'dataset_flujos_por_celda.pkl')

# # 3. Extraer los flujos únicos del día 01
# flujos_dia_clave = set(df_referencia['Clave_Flujo'].unique())
# flujos_dia_cluster = set(df_referencia['Flujo_Clusterizado'].unique())

# # 4. Filtrar 'dataset_celdas_por_flujo'
# # Nos quedamos solo con las filas cuyos flujos existen en el archivo del día 
# DF_Flujos = df_celdas_por_flujo_raw[
#     df_celdas_por_flujo_raw['Clave_Flujo'].isin(flujos_dia_clave)
# ].copy()

# # Eliminar celdas que no tengan flujos tras el filtro y guardar en DF_cells
# DF_cells = df_flujos_por_celda_raw[
#     df_flujos_por_celda_raw['Flujos_Clusterizados'].map(len) > 0
# ].copy()

# print(f"Las celdas con los flujos filtrados son: {DF_cells}")

# # Opcional: convertir de vuelta a string si necesitas el formato original para guardar
# # DF_cells['Flujos_Clusterizados'] = DF_cells['Flujos_Clusterizados'].astype(str)

# print(f"Filtrado listo: {len(DF_Flujos)} flujos y {len(DF_cells)} celdas activas para el día {fecha_sel}.")
# # FIN DEL BLOQUE DE FILTRADO -------------------------
# # DF_Flujos y DF_cells ya filtrados
# # DF_Flujos.to_csv(PATH_resultados + f'dataset_celdas_por_flujo_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.csv', index=False)
# # DF_cells.to_csv(PATH_resultados + f'dataset_flujos_por_celda_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.csv', index=False)

# # DF_Flujos.to_pickle(PATH_resultados + f'dataset_celdas_por_flujo_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.pkl')
# # DF_cells.to_pickle(PATH_resultados + f'dataset_flujos_por_celda_{dia_sel}_{hora_ini_str}_{hora_fin_str}_filtrado.pkl')
# #----------------------------------------------------------------------------------------------------

#%%
## Datos de complejidad----------------------------------------------------------------------------
# 
# 1. Cargar los datos
df_complejidad_sum = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Suma_{fecha_sel}_Completo.pkl')
df_complejidad_media = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Media_{fecha_sel}_Diaria.pkl')

# 2. Cambiar el nombre de la columna 'celda' a 'Cell_Name'
df_complejidad_sum = df_complejidad_sum.rename(columns={'Celda': 'Cell_Name'})
df_complejidad_media = df_complejidad_media.rename(columns={'Celda': 'Cell_Name'})

## Calcular el Z-score de la complejidad para cada celda----------------------------------------------------------------------------

# (Asumiendo que la columna de valores numéricos se llama 'Complejidad_Total_Dia')
media_comp = df_complejidad_sum['Complejidad_Total_Dia'].mean()
std_comp = df_complejidad_sum['Complejidad_Total_Dia'].std()

df_complejidad_sum['z_score'] = (df_complejidad_sum['Complejidad_Total_Dia'] - media_comp) / std_comp

media_comp_m = df_complejidad_media['Media_Complejidad'].mean()
std_comp_m = df_complejidad_media['Media_Complejidad'].std()

df_complejidad_media['z_score'] = (df_complejidad_media['Media_Complejidad'] - media_comp_m) / std_comp_m


### Complejidad de horas---------------------------------------------
# Para el análisis de una franja horaria
dfs_horas = []# Lista para almacenar los DataFrames de cada hora

hora_inicio_busqueda = int(hora_ini_str)
hora_fin_busqueda = int(hora_fin_str)

if hora_fin_busqueda == 0:
    hora_fin_busqueda = 24

print("Buscando y leyendo archivos de complejidad...")

# 3. Recorrer los archivos en el directorio
for archivo in os.listdir(PATH_COMPLEJIDAD_DIA):
    # Buscamos archivos que cumplan con el patrón del día seleccionado
    if archivo.startswith(f"Complejidad_por_hora_{fecha_sel}_") and (archivo.endswith('.pkl') or archivo.endswith('.csv')):
        
        # Extraemos las horas usando expresiones regulares (ej. extrae '08' y '09' de '08-09')
        # El \.csv$ o \.pkl$ asegura que busque el patrón pegado al final del archivo
        match = re.search(r'(\d{2})-(\d{2})\.(pkl)$', archivo)
        if match:
            h_ini_archivo = int(match.group(1))
            h_fin_archivo = int(match.group(2))
            if h_fin_archivo == 0:
                h_fin_archivo = 24
            
            print(f'evaluando el archivo de hora: {h_ini_archivo}-{h_fin_archivo}')

            # Filtramos: comprobar si el archivo cae dentro del rango de 8 a 12
            if h_ini_archivo >= hora_inicio_busqueda and h_fin_archivo <= hora_fin_busqueda:
                path_archivo = os.path.join(PATH_COMPLEJIDAD_DIA, archivo)                
                print(f" -> Cargando: {archivo}")
                
                # Leer según la extensión del archivo
                if archivo.endswith('.pkl'):
                    df_hora = pd.read_pickle(path_archivo)            
                # Guardamos el DataFrame en nuestra lista
                dfs_horas.append(df_hora)

# 4. Calcular la complejidad por hora y la media para cada celda
if dfs_horas:
    # Empezamos utilizando el primer DataFrame como base (solo con la columna Celda)
    df_final = dfs_horas[0][["Celda"]].copy()

    # Combinamos todos los dataframes de la lista añadiendo una columna por cada tramo horario
    for i, df_hora in enumerate(dfs_horas):
        # Creamos un nombre dinámico para la columna (ej: Compl_Hora_1, Compl_Hora_2...)
        # Si prefieres mapearlo con el nombre real del archivo (ej: 08-09), se puede adaptar.
        nombre_columna_hora = f"Compl_Hora_{i+1}"
        print(f"el df_hora es: {df_hora}")
        print(f"el nombre de la columna es: {nombre_columna_hora}")

        # Renombramos la columna de complejidad temporalmente antes de fusionar
        df_temp = df_hora.rename(columns={"Suma_Complejidad_total": nombre_columna_hora})
        print(f"el df_temp es: {df_temp.columns}")

        # Fusionamos con el dataframe principal usando 'Celda' como clave común
        df_final = pd.merge(df_final, df_temp, on="Celda", how="outer")

    # Rellenamos posibles valores vacíos con 0 (por si alguna celda no aparece en alguna hora)
    df_final = df_final.fillna(0)

    # Seleccionamos solo las columnas de las horas para calcular la media de forma limpia
    columnas_horas = [col for col in df_final.columns if col.startswith("Compl_Hora_")]

    # Añadimos la columna final con la media exacta de las horas por cada fila (axis=1)
    df_final["Media_Complejidad"] = df_final[columnas_horas].mean(axis=1)

    print("\n¡Tabla de complejidad por celdas generada con éxito!")
    print(df_final.head())

    # 5. Guardar el resultado final (csv y pkl)
    h_ini_str = hora_ini.strftime("%H_%M_%S")
    h_fin_str = hora_fin.strftime("%H_%M_%S")

    df_final.rename(columns={'Celda': 'Cell_Name'}, inplace=True)

    df_final.to_csv(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}_{h_ini_str}_{h_fin_str}.csv",index=False)
    df_final.to_pickle(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}_{h_ini_str}_{h_fin_str}.pkl")

    df_comp_hora_med = df_final.copy()
else:
    print(f"No se encontraron archivos de complejidad para el rango {hora_inicio_busqueda}-{hora_fin_busqueda} en el día {fecha_sel}.")
#--------------------------------------------------------------------

df_complejidad_media = df_comp_hora_med.copy()

#%%
if not sab_tipo1.empty:
    df_sabs_t1_final= sab_tipo1.copy()
else:
    df_sabs_t1_final = pd.DataFrame()


#%%
## Para los del tipo 2 -----------------------------------------------------------

# Ya están diseñados previamente las celdas del SAB del tipo 2. 
# La lógica de selección de los sabs del tipo 2 es muy diferente a los del tipo, porque ha centrado más
# en el análisis de los conflictos entre los flujos y si se sitúan en zonas de frontera. Lo cual nunca estarán
# en zonas de baja complejidad y no tocarán a los SAB del tipo 1.

# Datos del tipo 2 con los diseños hechos
df_sabs_t2 = gpd.GeoDataFrame(sab_tipo2, geometry='forma de SAB', crs=gdf_mallado.crs)
tipo2_cells = df_sabs_t2['Celda'].tolist()

# Iteramos por cada fila del DataFrame
for index, row in df_sabs_t2.iterrows():
    nodo_id = row['nodo_id']
    poligono_final_t2 = row['forma de SAB']
    datos_flujos = row['Flujos de SAB']
    
    # Extraemos IDs y líneas de esta fila específica
    ids_flujos = [d['Original_ID'] for d in datos_flujos]
    lineas_flujos = [d['Line'] for d in datos_flujos]
    
    # Creamos una figura para cada nodo
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Dibujar flujos
    for linea in lineas_flujos:
        ax.plot(*linea.xy, color='steelblue', alpha=0.15, linewidth=1.5)
    
    # 2. Calcular y dibujar puntos de cruce (solo para esta fila)
    puntos_cruce = []
    for i in range(len(lineas_flujos)):
        for j in range(i + 1, len(lineas_flujos)):
            if lineas_flujos[i].intersects(lineas_flujos[j]):
                inter = lineas_flujos[i].intersection(lineas_flujos[j])
                if inter.geom_type == 'Point':
                    puntos_cruce.append(inter)
                elif inter.geom_type == 'MultiPoint':
                    puntos_cruce.extend(list(inter.geoms))
    
    for p in puntos_cruce:
        ax.scatter(p.x, p.y, color='red', s=5, alpha=0.6, zorder=5)

    # 3. Dibujar el Polígono de esta fila
    if poligono_final_t2:
        if poligono_final_t2.geom_type == 'Polygon':
            x_poly, y_poly = poligono_final_t2.exterior.xy
            ax.plot(x_poly, y_poly, color='magenta', linewidth=3, label=f'SAB Nodo {nodo_id}')
            ax.fill(x_poly, y_poly, color='magenta', alpha=0.1)
        elif poligono_final_t2.geom_type == 'MultiPolygon':
            for p in poligono_final_t2.geoms:
                x_p, y_p = p.exterior.xy
                ax.plot(x_p, y_p, color='magenta', linewidth=3)
                ax.fill(x_p, y_p, color='magenta', alpha=0.1)

    for index, row in DF_info_conf.iterrows():
        poly = row['Contorno Sector']
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5)
        ax.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
                fontsize=10, ha='center', weight='bold', alpha=0.6)
        
    gdf_mallado.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6)

    ax.set_title(f"Visualización SAB - Nodo {nodo_id}")
    ax.legend()
    plt.show()

#%% asignación de 1º tipo SAB
# ==================================================================================== #
# FUSIÓN AUTOMÁTICA Y REDISEÑO DINÁMICO DE SECTORES CON SAB TIPO 1
# ==================================================================================== #

print("\n--- INICIANDO PROCESO AUTOMÁTICO DE INTERCAMBIO DE SECTORES ---")

# Aseguramos que tenemos copias limpias para no sobreescribir el DataFrame original en cada iteración
df_sectores_base = DF_info_conf.copy()

# Iteramos sobre cada SAB Tipo 1 definitivo detectado en el día
if df_sabs_t1_final.empty:
    print("No se detectaron SAB Tipo 1 para el día seleccionado. No se realizará ningún intercambio.")


if not df_sabs_t1_final.empty:
    for idx_sab, sab_row in df_sabs_t1_final.iterrows():
        sab_geom = sab_row['geometry']
        flujo_id = sab_row['Flujo_ID']
        
        # 1. DETECTAR EN QUÉ SECTOR ESTÁ EL SAB (Sector Origen)
        # Buscamos el sector que contiene mayoritariamente o interseca con el SAB
        sector_origen_row = None
        for idx_s, s_row in df_sectores_base.iterrows():
            if s_row['Contorno Sector'].intersects(sab_geom):
                # Verificamos que realmente comparta un área significativa (por si toca bordes)
                if s_row['Contorno Sector'].intersection(sab_geom).area > 0.5:
                    sector_origen_row = s_row
                    break
                    
        if sector_origen_row is None:
            print(f"Aviso: No se encontró sector de origen para el SAB del flujo {flujo_id}.")
            continue
            
        id_origen = sector_origen_row['SECTOR_ID']
        poly_origen = sector_origen_row['Contorno Sector']
        
        # 2. DETECTAR SECTORES INTERCAMBIABLES (Vecinos que tocan el SAB)
        sectores_destino = []
        for idx_s, s_row in df_sectores_base.iterrows():
            id_dest = s_row['SECTOR_ID']
            poly_dest = s_row['Contorno Sector']
            
            # Si es un sector diferente pero toca al SAB, es un candidato intercambiable
            if id_dest != id_origen and poly_dest.intersects(sab_geom):
                sectores_destino.append(id_dest)
                
        print(f"\n> Procesando Flujo SAB: {flujo_id}")
        print(f"  - Sector origen detectado: {id_origen}")
        print(f"  - Sectores vecinos intercambiables detectados: {sectores_destino}")
        
        # 3. BUCLE PARA CADA SECTOR DESTINO DETECTADO: Simular el traspaso/intercambio
        for id_destino in sectores_destino:
            print(f"    * Simulando transferencia [{id_origen} -> {id_destino}]...")
            
            # Recuperamos el polígono del sector destino actual
            poly_destino = df_sectores_base[df_sectores_base['SECTOR_ID'] == id_destino]['Contorno Sector'].iloc[0]
            
            # OPERACIONES GEOMÉTRICAS AUTOMÁTICAS:
            # A. Juntar el SAB al sector de destino (Unión)
            fusion_destino_sab = poly_destino.union(sab_geom)
            
            # B. Quitar la parte del SAB del sector de origen (Diferencia)
            fusion_origen_restado = poly_origen.difference(sab_geom)
            
            # 4. ILUSTRACIÓN EN EL MAPA DINÁMICO
            fig, ax_auto = plt.subplots(figsize=(12, 10))
            
            # Fondo: Mallado del ACC
            if not gdf_mallado.empty:
                gdf_mallado.plot(ax=ax_auto, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.4)
                
            # Dibujar el resto de sectores que no se alteran en esta simulación
            for _, s_row in df_sectores_base.iterrows():
                s_id = s_row['SECTOR_ID']
                if s_id != id_origen and s_id != id_destino:
                    x, y = s_row['Contorno Sector'].exterior.xy
                    ax_auto.fill(x, y, alpha=0.3, edgecolor='black', facecolor='whitesmoke', linewidth=1)
                    # ax_auto.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
                    #              s_id, fontsize=9, ha='center', weight='bold', color='gray')
            
            # Convertir a GeoDataFrames temporales para graficar con el CRS correcto
            gdf_origen_mod = gpd.GeoDataFrame(geometry=[fusion_origen_restado], crs=gdf_mallado.crs)
            gdf_destino_mod = gpd.GeoDataFrame(geometry=[fusion_destino_sab], crs=gdf_mallado.crs)
            
            # Pintar los dos sectores modificados
            gdf_origen_mod.plot(ax=ax_auto, color='tomato', alpha=0.6, edgecolor='darkred', linewidth=2)
            gdf_destino_mod.plot(ax=ax_auto, color='cyan', alpha=0.6, edgecolor='blue', linewidth=2)
            
            # Colocar etiquetas de texto en sus nuevos centros representativos
            if not fusion_origen_restado.is_empty:
                p_origen = gdf_origen_mod.geometry.representative_point().iloc[0]
                # ax_auto.text(p_origen.x, p_origen.y, f"{id_origen}", 
                #              fontsize=11, fontweight='bold', color='darkred', ha='center', va='center')
                            
            if not fusion_destino_sab.is_empty:
                p_destino = gdf_destino_mod.geometry.representative_point().iloc[0]
                # ax_auto.text(p_destino.x, p_destino.y, f"{id_destino}\n", 
                #              fontsize=11, fontweight='bold', color='blue', ha='center', va='center')
            
            # Dibujar las líneas de flujos de fondo para dar contexto al analista
            for _, f_row in DF_Flujos.iterrows():
                nombre_f = f_row['Flujo_Clusterizado']
                x_f, y_f = f_row['Line'].xy
                if nombre_f == flujo_id:
                    # El flujo que provocó este SAB se destaca fuertemente
                    ax_auto.plot(x_f, y_f, color='magenta', linewidth=3, zorder=5)
                    # ax_auto.text(x_f[len(x_f)//2], y_f[len(y_f)//2], f"Flujo: {nombre_f}", 
                    #              color='purple', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, pad=1))
                else:
                    ax_auto.plot(x_f, y_f, color='gray', linewidth=0.8, alpha=0.3, zorder=2)
                    
            # Leyenda manual dinámica
            custom_handles = [
                Patch(facecolor='tomato', edgecolor='darkred', alpha=0.6, label=f'{id_origen}'),
                Patch(facecolor='cyan', edgecolor='blue', alpha=0.6, label=f'{id_destino}'),
                Line2D([0], [0], color='magenta', lw=3, label=f'Flujo objetivo')
            ]
            ax_auto.legend(handles=custom_handles, loc='upper right', framealpha=0.9)
            
            # Detalles finales de la gráfica
            ax_auto.set_title(f"Asignación de SAB a Sector {id_destino}\n {fecha_sel}  {hora_ini_str}h - {hora_fin_str}h", fontsize=12)
            ax_auto.set_xlabel('Longitud [º]')
            ax_auto.set_ylabel('Latitud [º]')
            ax_auto.set_aspect('equal')
            
            plt.tight_layout()
            plt.show()

#%% asignación de 2º tipo SAB y la complejidad inicial
# ==================================================================================== #
# FUSIÓN AUTOMÁTICA Y REDISEÑO DINÁMICO DE SECTORES CON SAB TIPO 2
# ==================================================================================== #

print("\n--- INICIANDO PROCESO AUTOMÁTICO DE INTERCAMBIO DE SECTORES PARA SAB TIPO 2 ---")

# Copia limpia de los sectores base para las simulaciones
df_sectores_base_t2 = DF_info_conf.copy()

# Iteramos sobre cada fila de los SAB Tipo 2 detectados
for idx_sab2, sab2_row in df_sabs_t2.iterrows():
    sab_geom = sab2_row['forma de SAB']
    nodo_id = sab2_row['nodo_id']
    datos_flujos = sab2_row['Flujos de SAB'] # Lista de diccionarios con flujos implicados
    
    # Extraemos los IDs de los flujos críticos de este SAB específico
    flujos_criticos_ids = [d['Original_ID'] for d in datos_flujos]
    
    if sab_geom is None or sab_geom.is_empty:
        continue

    # 1. DETECTAR EN QUÉ SECTOR ESTÁ EL SAB TIPO 2 (Sector Origen)
    # Evaluamos qué sector interseca mayoritariamente con la geometría del SAB
    sector_origen_row = None
    max_area_interseccion = 0
    
    for idx_s, s_row in df_sectores_base_t2.iterrows():
        poly_sector = s_row['Contorno Sector']
        if poly_sector.intersects(sab_geom):
            area_int = poly_sector.intersection(sab_geom).area
            # Nos quedamos con el sector que contenga la mayor parte del SAB como origen
            if area_int > max_area_interseccion:
                max_area_interseccion = area_int
                sector_origen_row = s_row
                
    if sector_origen_row is None or max_area_interseccion < 1e-5:
        print(f"Aviso: No se encontró sector de origen claro para el SAB del Nodo {nodo_id}.")
        continue
        
    id_origen = sector_origen_row['SECTOR_ID']
    poly_origen = sector_origen_row['Contorno Sector']
    
    # 2. DETECTAR SECTORES INTERCAMBIABLES (Sectores adyacentes/vecinos que tocan el SAB)
    sectores_destino = []
    poly_total = poly_origen  # Lista para almacenar polígonos de sectores destino para análisis posterior

    for idx_s, s_row in df_sectores_base_t2.iterrows():
        id_dest = s_row['SECTOR_ID']
        poly_dest = s_row['Contorno Sector']
        
        # Si es un sector vecino diferente que interseca significativamente con el SAB
        if id_dest != id_origen and poly_dest.intersects(sab_geom):
            if poly_dest.intersection(sab_geom).area > 1e-5:
                sectores_destino.append(id_dest)
                poly_total = poly_total.union(poly_dest)  # Guardamos el polígono del sector destino para análisis posterior

        
    print(f"\n> Procesando SAB Tipo 2 - Nodo ID: {nodo_id}")
    print(f"  - Sector origen detectado: {id_origen}")
    print(f"  - Sectores vecinos intercambiables detectados: {sectores_destino}")
    
    # 3. BUCLE PARA CADA SECTOR DESTINO DETECTADO: Simular transferencia espacial
    for id_destino in sectores_destino:
        print(f"    * Simulando transferencia del SAB [{id_origen} -> {id_destino}]...")
        
        # Recuperamos el polígono limpio del sector destino asignado
        poly_destino = df_sectores_base_t2[df_sectores_base_t2['SECTOR_ID'] == id_destino]['Contorno Sector'].iloc[0]
        
        # OPERACIONES GEOMÉTRICAS DE TRASPASO AUTOMÁTICO:
        # A. Sumar el SAB al sector que lo absorbe (Unión)
        fusion_destino_sab = poly_destino.union(sab_geom)
        
        # B. Quitar la porción de SAB del sector donde residía (Diferencia)
        fusion_origen_restado = poly_origen.difference(sab_geom)
        
        # 4. ILUSTRACIÓN GRÁFICA DEL MAPA DINÁMICO
        fig, ax_auto2 = plt.subplots(figsize=(12, 10))
        
        # Fondo: Mallado del ACC tenue
        if not gdf_mallado.empty:
            gdf_mallado.plot(ax=ax_auto2, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.4)
            
        # Dibujar todos los sectores estables (los que no participan en este intercambio)
        for _, s_row in df_sectores_base_t2.iterrows():
            s_id = s_row['SECTOR_ID']
            if s_id != id_origen and s_id != id_destino:
                x, y = s_row['Contorno Sector'].exterior.xy
                ax_auto2.fill(x, y, alpha=0.3, edgecolor='black', facecolor='whitesmoke', linewidth=1)
                ax_auto2.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
                             s_id, fontsize=9, ha='center', weight='bold', color='gray')
        
        # Empaquetar geometrías en GeoDataFrames temporales para respetar proyecciones
        gdf_origen_mod = gpd.GeoDataFrame(geometry=[fusion_origen_restado], crs=gdf_mallado.crs)
        gdf_destino_mod = gpd.GeoDataFrame(geometry=[fusion_destino_sab], crs=gdf_mallado.crs)
        
        # Renderizar sectores modificados con colores distintivos
        gdf_origen_mod.plot(ax=ax_auto2, color='tomato', alpha=0.6, edgecolor='darkred', linewidth=2)
        gdf_destino_mod.plot(ax=ax_auto2, color='cyan', alpha=0.6, edgecolor='blue', linewidth=2)
        
        # Colocar etiquetas en los nuevos centroides representativos pos-fusión
        if not fusion_origen_restado.is_empty:
            p_origen = gdf_origen_mod.geometry.representative_point().iloc[0]
            ax_auto2.text(p_origen.x, p_origen.y, f"{id_origen}", 
                          fontsize=11, fontweight='bold', color='darkred', ha='center', va='center')
                         
        if not fusion_destino_sab.is_empty:
            p_destino = gdf_destino_mod.geometry.representative_point().iloc[0]
            ax_auto2.text(p_destino.x, p_destino.y, f"{id_destino}", 
                          fontsize=11, fontweight='bold', color='blue', ha='center', va='center')
        
        f_rep = []
        # Representación de las líneas de flujos
        for _, f_row in DF_Flujos.iterrows():
            nombre_f = f_row['Flujo_Clusterizado']
            line_f = f_row['Line']
            x_f, y_f = f_row['Line'].xy
            sector_f = f_row['Sector']
            
            if line_f.intersects(poly_total):
                # Si el flujo actual pertenece al conjunto de flujos conflictivos de este SAB Tipo 2            
                ax_auto2.plot(x_f, y_f, color='magenta', linewidth=0.5, zorder=5, linestyle='-')
                f_rep.append(nombre_f)
                # Ponemos una etiqueta de texto discreta en la mitad del trayecto del flujo
                # ax_auto2.text(x_f[len(x_f)//2], y_f[len(y_f)//2], nombre_f, 
                #              color='purple', fontsize=8, weight='bold', alpha=0.8,
                #              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
                
        # Leyenda dinámica adaptada
        custom_handles_t2 = [
            Patch(facecolor='tomato', edgecolor='darkred', alpha=1, label=f'Sector {id_origen}'),
            Patch(facecolor='cyan', edgecolor='blue', alpha=1, label=f'Sector {id_destino}'),
            Line2D([0], [0], color='magenta', lw=2, label=f'Flujos')
        ]
        ax_auto2.legend(handles=custom_handles_t2, loc='upper right', framealpha=0.9)
        
        # Títulos e información final del eje
        ax_auto2.set_title(f"Asignación de SAB al sector {id_destino}\n fecha: {fecha_sel}  {hora_ini_str}h - {hora_fin_str}h", fontsize=12)
        ax_auto2.set_xlabel('Longitud [º]')
        ax_auto2.set_ylabel('Latitud [º]')
        ax_auto2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()


# # ==================================================================================== #
# # COMPLEJIDAD INICIAL POR SECTOR (ESTADO BASE USANDO ASIGNACIÓN DIRECTA)
# # ==================================================================================== #

# print("\n--- CALCULANDO Y REPRESENTANDO LA COMPLEJIDAD INICIAL POR SECTOR ---")

# # Preparar la figura general
# fig, ax_inicial = plt.subplots(figsize=(14, 11))

# # Fondo: Mallado base muy tenue
# if not gdf_mallado.empty:
#     gdf_mallado.plot(ax=ax_inicial, color='whitesmoke', edgecolor='lightgray', linewidth=0.2, alpha=0.5)

# # Paleta de colores para diferenciar los sectores iniciales
# sectores_unicos = DF_info_conf['SECTOR_ID'].unique()
# cmap_sectores = cm.get_cmap('tab20', len(sectores_unicos))
# custom_handles_inicial = []

# # Iterar por todos los sectores usando la columna 'Sector' del mallado original
# for idx_s, s_id in enumerate(sectores_unicos):
#     color_sector = cmap_sectores(idx_s)
    
#     # Extraer directamente las celdas que ya pertenecen a este sector
#     celdas_sector_gdf = gdf_mallado[gdf_mallado['Sector'] == s_id]
    
#     if not celdas_sector_gdf.empty:
#         celdas_en_sector = celdas_sector_gdf['Cell_Name'].values
        
#         # Calcular la suma de complejidad inicial cruzando con df_complejidad_media
#         comp_inicial_sector = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_sector)]['Media_Complejidad'].sum()
        
#         # Pintar las celdas del sector
#         celdas_sector_gdf.plot(ax=ax_inicial, color=color_sector, alpha=0.6, edgecolor='gray', linewidth=0.3)
        
#         # Colocar la etiqueta en el centroide de sus celdas
#         centroide = celdas_sector_gdf.geometry.unary_union.centroid
#         ax_inicial.text(
#             centroide.x, centroide.y, f"{s_id}\nCompl: {comp_inicial_sector:.2f}", 
#             fontsize=10, weight='bold', color='black', 
#             ha='center', va='center', 
#             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
#         )
        
#         custom_handles_inicial.append(Patch(facecolor=color_sector, edgecolor='gray', alpha=0.6, label=f"{s_id} (Inicial)"))

# # Pintar TODOS los flujos diarios en gris/azul tenue para dar contexto
# for _, f_row in DF_Flujos.iterrows():
#     x_f, y_f = f_row['Line'].xy
#     ax_inicial.plot(x_f, y_f, color='royalblue', linewidth=0.8, alpha=0.25, zorder=2)

# custom_handles_inicial.append(Line2D([0], [0], color='royalblue', lw=1, alpha=0.5, label='Flujos de tráfico (Contexto)'))

# ax_inicial.legend(handles=custom_handles_inicial, loc='upper right', framealpha=0.9, fontsize=9)
# ax_inicial.set_title(f"Estado Inicial: Distribución de Mallado y Complejidad por Sector\nConfiguración: {configuracion_estudio} | Fecha: {dia_sel}-{mes_sel}-2022", fontsize=14, weight='bold')
# ax_inicial.set_xlabel('Longitud [º]')
# ax_inicial.set_ylabel('Latitud [º]')
# ax_inicial.set_aspect('equal')

# plt.tight_layout()
# plt.show()

# print("--- MAPA INICIAL GENERADO CON ÉXITO. PROCEDIENDO A LAS SIMULACIONES INDIVIDUALES ---")

# ==================================================================================== #
# COMPLEJIDAD INICIAL POR SECTOR (ESTADO BASE ANTES DEL BUCLE)
# ==================================================================================== #

print("\n--- CALCULANDO Y REPRESENTANDO LA COMPLEJIDAD INICIAL POR SECTOR ---")

# Precalcular áreas originales del mallado
gdf_mallado['area_total'] = gdf_mallado.geometry.area

# Lista para recolectar las asignaciones de cada celda
registros_mapeo = []

# 1. EVALUAR INTERSECCIONES DE CELDAS CON TODOS LOS SECTORES
for idx_c, cell_row in gdf_mallado.iterrows():
    cell_name = cell_row['Cell_Name']
    cell_geom = cell_row['geometry']
    cell_area = cell_row['area_total']
    
    sectores_intersectados = []
    porcentajes = {}
    
    for idx_s, s_row in DF_info_conf.iterrows():
        s_id = s_row['SECTOR_ID']
        poly_sector = s_row['Contorno Sector']
        
        if cell_geom.intersects(poly_sector):
            area_int = cell_geom.intersection(poly_sector).area
            porcentaje_int = area_int / cell_area
            
            # Consideramos que pertenece o disputa el sector si la intersección es real (> 0.5%)
            if porcentaje_int > 0.5: 
                sectores_intersectados.append(s_id)
                porcentajes[s_id] = porcentaje_int
                
    # Determinar el estado de asignación y el sector principal
    if len(sectores_intersectados) == 0:
        sector_asignado = "Ninguno"
        estado = "Huérfana"
    elif len(sectores_intersectados) == 1:
        sector_asignado = sectores_intersectados[0]
        estado = "Único"
    else:
        # Si toca varios sectores, encontramos el que tiene mayor porcentaje de área
        sector_asignado = max(porcentajes, key=porcentajes.get)
        estado = "Coincidente"
        
    # Guardar registro de la celda
    registros_mapeo.append({
        'Cell_Name': cell_name,
        'Sectores_Intersec': sectores_intersectados,
        'Num_Sectores': len(sectores_intersectados),
        'Sector_Principal': sector_asignado,
        'Porcentajes_Area': porcentajes,
        'Estado': estado
    })

# Crear el DataFrame definitivo de mapeo
df_mapeo_celdas = pd.DataFrame(registros_mapeo)

# ==================================================================================== #
# REPRESENTACIÓN GRÁFICA Y CÁLCULO DE COMPLEJIDAD INICIAL
# ==================================================================================== #

fig, ax_inicial = plt.subplots(figsize=(14, 11))

# Fondo: Mallado base muy tenue
if not gdf_mallado.empty:
    gdf_mallado.plot(ax=ax_inicial, color='whitesmoke', edgecolor='lightgray', linewidth=0.2, alpha=0.5)

cmap_sectores = cm.get_cmap('tab20', len(DF_info_conf))
custom_handles_inicial = []

# Iterar por todos los sectores para pintar su mallado basado en el criterio clásico (>50%)
for idx_s, s_row in DF_info_conf.iterrows():
    s_id = s_row['SECTOR_ID']
    color_sector = cmap_sectores(idx_s)
    
    # Filtrar las celdas cuyo Sector Principal sea el actual y tengan >50% de área asignada
    celdas_del_sector = df_mapeo_celdas[
        (df_mapeo_celdas['Sector_Principal'] == s_id) & 
        (df_mapeo_celdas['Cell_Name'].apply(lambda x: df_mapeo_celdas[df_mapeo_celdas['Cell_Name'] == x]['Porcentajes_Area'].values[0].get(s_id, 0) > 0.5))
    ]['Cell_Name'].values
    
    celdas_sector_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_del_sector)]
    
    if not celdas_sector_gdf.empty:
        # Calcular complejidad inicial
        comp_inicial_sector = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_del_sector)]['Media_Complejidad'].sum()
        
        # Pintar celdas estándar del sector
        celdas_sector_gdf.plot(ax=ax_inicial, color=color_sector, alpha=0.6, edgecolor='gray', linewidth=0.3)
        
        # Etiqueta de complejidad
        centroide = celdas_sector_gdf.geometry.unary_union.centroid
        ax_inicial.text(
            centroide.x, centroide.y, f"{s_id}\nCompl: {comp_inicial_sector:.2f}", 
            fontsize=10, weight='bold', color='black', 
            ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )
        custom_handles_inicial.append(Patch(facecolor=color_sector, edgecolor='gray', alpha=0.6, label=f"{s_id} (Inicial)"))

# --- VISUALIZACIÓN EXCLUSIVA DE CELDAS COINCIDENTES EN EL MAPA ---
celdas_coincidentes_nombres = df_mapeo_celdas[df_mapeo_celdas['Estado'] == 'Coincidente']['Cell_Name'].values
gdf_coincidentes = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_coincidentes_nombres)]

if not gdf_coincidentes.empty:
    # Pintamos las celdas en disputa con un patrón de líneas (hatch) o un color rojo/alerta traslúcido
    gdf_coincidentes.plot(ax=ax_inicial, facecolor='none', edgecolor='red', linewidth=0.8, hatch='//', alpha=0.4, zorder=4)
    custom_handles_inicial.append(Patch(facecolor='none', edgecolor='red', hatch='//', alpha=0.6, label='Celdas Coincidentes (Disputa)'))

# Pintar flujos de tráfico
for _, f_row in DF_Flujos.iterrows():
    x_f, y_f = f_row['Line'].xy
    ax_inicial.plot(x_f, y_f, color='royalblue', linewidth=0.8, alpha=0.25, zorder=2)

custom_handles_inicial.append(Line2D([0], [0], color='royalblue', lw=1, alpha=0.5, label='Flujos de tráfico (Contexto)'))

ax_inicial.legend(handles=custom_handles_inicial, loc='upper right', framealpha=0.9, fontsize=9)
ax_inicial.set_title(f"Estado Inicial y Celdas Coincidentes por Sector\nConfiguración: {configuracion_estudio} | Fecha: {fecha_sel}", fontsize=14, weight='bold')
ax_inicial.set_xlabel('Longitud [º]')
ax_inicial.set_ylabel('Latitud [º]')
ax_inicial.set_aspect('equal')

plt.tight_layout()
plt.show()

print("--- MAPA INICIAL Y DATAFRAME DE ASIGNACIÓN GENERADOS ---")
print(f"Muestra del DataFrame `df_mapeo_celdas` (Total celdas coincidentes: {len(celdas_coincidentes_nombres)}):")
print(df_mapeo_celdas[['Cell_Name', 'Sectores_Intersec', 'Sector_Principal', 'Estado']].head())

# # Preparar la figura general
# fig, ax_inicial = plt.subplots(figsize=(14, 11))

# # Fondo: Mallado base muy tenue
# if not gdf_mallado.empty:
#     gdf_mallado.plot(ax=ax_inicial, color='whitesmoke', edgecolor='lightgray', linewidth=0.2, alpha=0.5)

# # Paleta de colores para diferenciar los sectores iniciales
# cmap_sectores = cm.get_cmap('tab20', len(DF_info_conf))
# custom_handles_inicial = []

# # Iterar por todos los sectores para calcular su mallado y complejidad inicial
# for idx_s, s_row in DF_info_conf.iterrows():
#     s_id = s_row['SECTOR_ID']
#     poly_sector = s_row['Contorno Sector']
#     color_sector = cmap_sectores(idx_s)
    
#     # Crear un GeoDataFrame temporal del sector actual
#     gdf_sector_temp = gpd.GeoDataFrame(geometry=[poly_sector], crs=gdf_mallado.crs)
    
#     # Intersección espacial para evaluar qué celdas le pertenecen originalmente
#     int_sector = gpd.overlay(gdf_mallado, gdf_sector_temp, how='intersection')
    
#     if not int_sector.empty:
#         # Filtrar por el criterio del >50% de su área dentro del sector
#         int_sector['porcentaje'] = int_sector.geometry.area / int_sector['area_total']
#         celdas_en_sector = int_sector[int_sector['porcentaje'] > 0.5]['Cell_Name'].values
        
#         # Extraer el GDF de celdas asignadas
#         celdas_sector_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_en_sector)]
        
#         # Calcular la suma de complejidad inicial para este sector
#         comp_inicial_sector = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_sector)]['Media_Complejidad'].sum()
        
#         # Pintar las celdas del sector en el mapa inicial
#         if not celdas_sector_gdf.empty:
#             celdas_sector_gdf.plot(ax=ax_inicial, color=color_sector, alpha=0.6, edgecolor='gray', linewidth=0.3)
            
#             # Colocar la etiqueta con el ID del sector y su complejidad en el centroide de sus celdas
#             centroide = celdas_sector_gdf.geometry.unary_union.centroid
#             ax_inicial.text(
#                 centroide.x, centroide.y, f"{s_id}\nCompl: {comp_inicial_sector:.2f}", 
#                 fontsize=10, weight='bold', color='black', 
#                 ha='center', va='center', 
#                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
#             )
            
#             # Añadir a la leyenda
#             custom_handles_inicial.append(Patch(facecolor=color_sector, edgecolor='gray', alpha=0.6, label=f"{s_id} (Inicial)"))

# # Pintar TODOS los flujos diarios en gris/azul tenue para dar contexto de las trayectorias originales
# for _, f_row in DF_Flujos.iterrows():
#     x_f, y_f = f_row['Line'].xy
#     ax_inicial.plot(x_f, y_f, color='royalblue', linewidth=0.8, alpha=0.25, zorder=2)

# custom_handles_inicial.append(Line2D([0], [0], color='royalblue', lw=1, alpha=0.5, label='Flujos de tráfico (Contexto)'))

# # Configuración estética del mapa inicial
# ax_inicial.legend(handles=custom_handles_inicial, loc='upper right', framealpha=0.9, fontsize=9)
# ax_inicial.set_title(f"Estado Inicial: Distribución de Mallado y Complejidad por Sector\nConfiguración: {configuracion_estudio} | Fecha: {dia_sel:02d}-{mes_sel:02d}-2022", fontsize=14, weight='bold')
# ax_inicial.set_xlabel('Longitud [º]')
# ax_inicial.set_ylabel('Latitud [º]')
# ax_inicial.set_aspect('equal')

# plt.tight_layout()
# plt.show()

# print("--- MAPA INICIAL GENERADO CON ÉXITO. PROCEDIENDO A LAS SIMULACIONES INDIVIDUALES ---")

#%% variación de complejidad al asignar de 2º tipo SAB
# ==================================================================================== #
# FUSIÓN AUTOMÁTICA, RE-MALLADO Y EVALUACIÓN DE COMPLEJIDAD PARA SAB TIPO 2
# ==================================================================================== #

print("\n--- INICIANDO PROCESO AUTOMÁTICO DE INTERCAMBIO, MALLADO Y COMPLEJIDAD (SAB T2) ---")

# Copia limpia de los sectores base para las simulaciones
df_sectores_base_t2 = DF_info_conf.copy()

# Precalcular áreas originales del mallado para optimizar el bucle
gdf_mallado['area_total'] = gdf_mallado.geometry.area

# Iteramos sobre cada fila de los SAB Tipo 2 detectados
for idx_sab2, sab2_row in df_sabs_t2.iterrows():
    sab_geom = sab2_row['forma de SAB']
    nodo_id = sab2_row['nodo_id']
    datos_flujos = sab2_row['Flujos de SAB'] 
    
    flujos_criticos_ids = [d['Original_ID'] for d in datos_flujos]
    
    if sab_geom is None or sab_geom.is_empty:
        continue

    # 1. DETECTAR EN QUÉ SECTOR ESTÁ EL SAB TIPO 2 (Sector Origen)
    sector_origen_row = None
    max_area_interseccion = 0
    
    for idx_s, s_row in df_sectores_base_t2.iterrows():
        poly_sector = s_row['Contorno Sector']
        if poly_sector.intersects(sab_geom):
            area_int = poly_sector.intersection(sab_geom).area
            if area_int > max_area_interseccion:
                max_area_interseccion = area_int
                sector_origen_row = s_row
                
    if sector_origen_row is None or max_area_interseccion < 1e-5:
        print(f"Aviso: No se encontró sector de origen claro para el SAB del Nodo {nodo_id}.")
        continue
        
    id_origen = sector_origen_row['SECTOR_ID']
    poly_origen = sector_origen_row['Contorno Sector']
    
    # 2. DETECTAR SECTORES INTERCAMBIABLES
    sectores_destino = []
    for idx_s, s_row in df_sectores_base_t2.iterrows():
        id_dest = s_row['SECTOR_ID']
        poly_dest = s_row['Contorno Sector']
        
        if id_dest != id_origen and poly_dest.intersects(sab_geom):
            if poly_dest.intersection(sab_geom).area > 1e-5:
                sectores_destino.append(id_dest)
                
    print(f"\n> Procesando SAB Tipo 2 - Nodo ID: {nodo_id}")
    print(f"  - Sector origen detectado: {id_origen}")
    print(f"  - Sectores vecinos intercambiables detectados: {sectores_destino}")
    # =========================================================================
    # --- SECCIÓN: EVALUACIÓN CASO CONSERVACIÓN (SAB EN ORIGEN + VECINOS) ---
    # =========================================================================
    print(f"    * Simulando CONSERVACIÓN del SAB en el sector original [{id_origen}]...")
    
    # --- 1. DETERMINAR LAS CELDAS QUE SE QUEDA EL ORIGEN TRAS CONSERVACIÓN ---
    # Una celda se queda en el origen si su Sector_Principal es id_origen 
    # O si es una celda "Coincidente" que intersecta al sector original (Prioridad Absoluta)
    celdas_en_origen_cons = df_mapeo_celdas[
        (df_mapeo_celdas['Sector_Principal'] == id_origen) | 
        ((df_mapeo_celdas['Estado'] == 'Coincidente') & 
         (df_mapeo_celdas['Sectores_Intersec'].apply(lambda x: id_origen in x)))
    ]['Cell_Name'].values

    celda_origen_cons_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_en_origen_cons)]
    print(f"      -> Celdas asignadas al origen {id_origen} (incluyendo coincidentes forzadas): {len(celdas_en_origen_cons)}")
    
    # Calcular complejidad del origen
    col_comp = 'Media_Complexity' if 'Media_Complexity' in df_complejidad_media.columns else 'Media_Complejidad'
    comp_origen_cons = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_origen_cons)][col_comp].sum()
    print(f"      -> Nueva complejidad acumulada (Conservación) {id_origen}: {comp_origen_cons:.2f}")
    
    # Diccionario para guardar celdas y complejidad remanente de vecinos
    adyacentes_cons_info = {}
    
    # --- 2. CALCULAR COMPLEJIDAD REMANENTE DE VECINOS USANDO EL DF_MAPEO ---
    for id_vecino in sectores_destino:
        # El vecino inicialmente reclama las celdas donde es Sector_Principal
        celdas_potenciales_vecino = df_mapeo_celdas[df_mapeo_celdas['Sector_Principal'] == id_vecino]['Cell_Name'].values
        
        # FILTRO CRÍTICO: Eliminar explícitamente cualquier celda que ya haya absorbido el origen 
        # (Esto resuelve de golpe las celdas coincidentes/en disputa y las remueve del adyacente)
        celdas_en_vecino = [c for c in celdas_potenciales_vecino if c not in celdas_en_origen_cons]
        
        print(f"      -> Celdas asignadas al vecino {id_vecino} tras excluir conflicto con origen: {len(celdas_en_vecino)}")
        
        celda_vecino_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_en_vecino)]
        comp_vecino = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_vecino)][col_comp].sum()
        
        print(f"      -> Complejidad remanente sector adyacente {id_vecino}: {comp_vecino:.2f}")
        
        adyacentes_cons_info[id_vecino] = {
            'gdf': celda_vecino_gdf,
            'complejidad': comp_vecino
        }

    # =========================================================================
    # --- 3. GRÁFICO PARA EL CASO CONSERVACIÓN INTEGRAL (Sin cambios estéticos) ---
    # =========================================================================
    fig, ax_cons = plt.subplots(figsize=(12, 10))
    elementos_leyenda = []
    if not gdf_mallado.empty:
        gdf_mallado.plot(ax=ax_cons, color='whitesmoke', edgecolor='lightgray', linewidth=0.2, alpha=0.5)
        
    for _, s_row in df_sectores_base_t2.iterrows():
        s_id = s_row['SECTOR_ID']
        if s_id != id_origen and s_id not in sectores_destino:
            x, y = s_row['Contorno Sector'].exterior.xy
            ax_cons.fill(x, y, alpha=0.15, edgecolor='black', facecolor='whitesmoke', linewidth=0.8)
            # ax_cons.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
            #              s_id, fontsize=9, ha='center', weight='bold', color='gray', alpha=0.5)
                         
    if not celda_origen_cons_gdf.empty:
        celda_origen_cons_gdf.plot(ax=ax_cons, color='forestgreen', alpha=0.6, edgecolor='darkgreen', linewidth=0.5)
        centroide_cons = celda_origen_cons_gdf.geometry.unary_union.centroid
        elementos_leyenda.append(
        mpatches.Patch(
            color='forestgreen', 
            alpha=0.6, 
            label=f"{id_origen}: {comp_origen_cons:.2f}"
        )
    )
        # ax_cons.text(
        #     centroide_cons.x, centroide_cons.y, f"{id_origen}\nCompl: {comp_origen_cons:.2f}", 
        #     fontsize=11, weight='bold', color='darkgreen', ha='center', va='center', 
        #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
        # )
        
    colores_vecinos = ['gold', 'orange', 'khaki']
    for idx_v, (id_vecino, info) in enumerate(adyacentes_cons_info.items()):
        v_gdf = info['gdf']
        v_comp = info['complejidad']
        if not v_gdf.empty:
            color_p = colores_vecinos[idx_v % len(colores_vecinos)]
            v_gdf.plot(ax=ax_cons, color=color_p, alpha=0.5, edgecolor='darkgoldenrod', linewidth=0.5)

            centroide_v = v_gdf.geometry.unary_union.centroid
            elementos_leyenda.append(
            mpatches.Patch(
                color=color_p, 
                alpha=0.5, 
                label=f"{id_vecino}: {v_comp:.2f}"
            ))

            # ax_cons.text(
            #     centroide_v.x, centroide_v.y, f"{id_vecino}\nCompl: {v_comp:.2f}", 
            #     fontsize=10, weight='bold', color='darkgoldenrod', ha='center', va='center', 
            #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            # )
        
    # for _, f_row in DF_Flujos.iterrows():
    #     nombre_f = f_row['Flujo_Clusterizado']
    #     x_f, y_f = f_row['Line'].xy
    #     if nombre_f in f_rep:
    #         ax_cons.plot(x_f, y_f, color='magenta', linewidth=1.5, zorder=5, linestyle='-')
    #     else:
    #         ax_cons.plot(x_f, y_f, color='gray', linewidth=0.6, alpha=0.15, zorder=2)
            
    ax_cons.legend(handles=elementos_leyenda, loc='upper right',frameon=True, facecolor='white', edgecolor='gray')
    ax_cons.set_title(f"Complejidad de los sectores iniciales \n {fecha_sel} {hora_ini_str}h - {hora_fin_str}h", fontsize=12)
    ax_cons.set_xlabel('Longitud [º]')
    ax_cons.set_ylabel('Latitud [º]')
    ax_cons.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    
    # 3. BUCLE PARA CADA SECTOR DESTINO DETECTADO
    for id_destino in sectores_destino:
        print(f"    * Simulando transferencia del SAB [{id_origen} -> {id_destino}]...")
        
        poly_destino = df_sectores_base_t2[df_sectores_base_t2['SECTOR_ID'] == id_destino]['Contorno Sector'].iloc[0]
        
        # OPERACIONES GEOMÉTRICAS DE TRASPASO:
        fusion_destino_sab = poly_destino.union(sab_geom)
        fusion_origen_restado = poly_origen.difference(sab_geom)
        
        # --- NUEVA SECCIÓN: RE-MALLADO DINÁMICO Y SUMA DE COMPLEJIDAD ---
        
        # Crear GeoDataFrames auxiliares para los dos sectores modificados
        gdf_origen_temporal = gpd.GeoDataFrame(geometry=[fusion_origen_restado], crs=gdf_mallado.crs)
        gdf_destino_temporal = gpd.GeoDataFrame(geometry=[fusion_destino_sab], crs=gdf_mallado.crs)
        
        # Intersección espacial para calcular el porcentaje de área oculta en cada sector modificado
        int_origen = gpd.overlay(gdf_mallado, gdf_origen_temporal, how='intersection')
        int_destino = gpd.overlay(gdf_mallado, gdf_destino_temporal, how='intersection')
        
        # Filtrar celdas asignadas al origen (>50% de su área dentro)
        int_origen['porcentaje'] = int_origen.geometry.area / int_origen['area_total']
        celdas_en_origen = int_origen[int_origen['porcentaje'] > 0.5]['Cell_Name'].values
        celda_origen_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_en_origen)]
        
        # Filtrar celdas asignadas al destino (>50% de su área dentro)
        int_destino['porcentaje'] = int_destino.geometry.area / int_destino['area_total']
        celdas_en_destino = int_destino[int_destino['porcentaje'] > 0.5]['Cell_Name'].values
        celda_destino_gdf = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_en_destino)]
        
        # Calcular las sumas de complejidad cruzando los nombres de celda válidos con df_complejidad_media
        comp_origen = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_origen)]['Media_Complejidad'].sum()
        comp_destino = df_complejidad_media[df_complejidad_media['Cell_Name'].isin(celdas_en_destino)]['Media_Complejidad'].sum()
        
        print(f"      -> Nueva complejidad acumulada {id_origen}: {comp_origen:.2f}")
        print(f"      -> Nueva complejidad acumulada {id_destino}: {comp_destino:.2f}")
        
        # 4. ILUSTRACIÓN GRÁFICA DEL MAPA DINÁMICO
        fig, ax_auto2 = plt.subplots(figsize=(12, 10))
        elementos_leyenda = []
        # Fondo: Mallado base del ACC muy tenue
        if not gdf_mallado.empty:
            gdf_mallado.plot(ax=ax_auto2, color='whitesmoke', edgecolor='lightgray', linewidth=0.2, alpha=0.5)
            
        # Dibujar todos los sectores estables de fondo
        for _, s_row in df_sectores_base_t2.iterrows():
            s_id = s_row['SECTOR_ID']
            if s_id != id_origen and s_id != id_destino:
                x, y = s_row['Contorno Sector'].exterior.xy
                ax_auto2.fill(x, y, alpha=0.15, edgecolor='black', facecolor='whitesmoke', linewidth=0.8)
                ax_auto2.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
                             s_id, fontsize=9, ha='center', weight='bold', color='gray', alpha=0.5)
        
        # Pintar el nuevo mallado asignado a cada sector modificado en lugar del polígono crudo
        if not celda_origen_gdf.empty:
            celda_origen_gdf.plot(ax=ax_auto2, color='tomato', alpha=0.6, edgecolor='darkred', linewidth=0.5)
            elementos_leyenda.append(
            mpatches.Patch(
                color='tomato', 
                alpha=0.6, 
                label=f"{id_origen}: {comp_origen:.2f}"
            ))
        if not celda_destino_gdf.empty:
            celda_destino_gdf.plot(ax=ax_auto2, color='cyan', alpha=0.6, edgecolor='blue', linewidth=0.5)
            elementos_leyenda.append(
            mpatches.Patch(
                color='cyan', 
                alpha=0.6, 
                label=f"{id_destino}: {comp_destino:.2f}"
            ))
        
        # # Función interna para colocar la etiqueta del total de complejidad en el centroide ponderado
        # def etiquetar_sector_comp(gdf, valor, id_sec, ax, color_texto):
        #     if not gdf.empty:
        #         centroide = gdf.geometry.unary_union.centroid
        #         ax.text(
        #             centroide.x, centroide.y, f"{id_sec}", 
        #             fontsize=11, weight='bold', color=color_texto, 
        #             ha='center', va='center'
        #         )

        # # Añadir las etiquetas con los resultados matemáticos exactos en el mapa
        # etiquetar_sector_comp(celda_origen_gdf, comp_origen, id_origen, ax_auto2, 'darkred')
        # etiquetar_sector_comp(celda_destino_gdf, comp_destino, id_destino, ax_auto2, 'blue')
        
        # Representación de las líneas de flujos críticos
        # for _, f_row in DF_Flujos.iterrows():
        #     nombre_f = f_row['Flujo_Clusterizado']
        #     x_f, y_f = f_row['Line'].xy
            
        #     if nombre_f in f_rep:
        #         ax_auto2.plot(x_f, y_f, color='magenta', linewidth=0.5, zorder=5, linestyle='-')
                
        #     else:
        #         ax_auto2.plot(x_f, y_f, color='gray', linewidth=0.6, alpha=0.15, zorder=2)

              
        # Leyenda dinámica adaptada
        ax_auto2.legend(handles=elementos_leyenda, loc='upper right',frameon=True, facecolor='white', edgecolor='gray')
        ax_auto2.set_title(f"Asignación de SAB al sector {id_destino}\n fecha: {fecha_sel} {hora_ini_str}h - {hora_fin_str}h", fontsize=12)
        ax_auto2.set_xlabel('Longitud [º]')
        ax_auto2.set_ylabel('Latitud [º]')
        ax_auto2.set_aspect('equal')
        # 
        plt.tight_layout()
        plt.show()

#%%
# ---------------------------------------------------------------------------------------- #
# -------------------------------- REPRESENTACIÓN GRÁFICA -------------------------------- #
# ---------------------------------------------------------------------------------------- #

print("\n--- GENERANDO REPRESENTACIÓN GRÁFICA ---")

fig, ax6 = plt.subplots(figsize=(12, 10))

# 1. Dibujar los Sectores del ACC (SIN etiqueta 'label')
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    ax6.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5)
    ax6.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6)

# 2. Dibujar el fondo: Todo el mallado del ACC
if not gdf_mallado.empty:
    gdf_mallado.plot(ax=ax6, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6)

# 3. Dibujar SABs Finales
if not df_sabs_t1_final.empty:
    # Eliminado el figsize interno que era redundante
    df_sabs_t1_final.plot(ax=ax6, color='cyan', alpha=0.5, edgecolor='blue')


# 4. Crear LEYENDA MANUAL
custom_lines_6 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=0.5, label='Forma de SAB del tipo 1')
]
ax6.legend(handles=custom_lines_6, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax6.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel}")
ax6.set_xlabel('Longitud [º]')
ax6.set_ylabel('Latitud [º]')
ax6.set_aspect('equal')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------

## SAB finales
fig, ax8 = plt.subplots(figsize=(12, 10))

# 1. Dibujar los Sectores del ACC (SIN etiqueta 'label')
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    ax8.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5)
    ax8.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6)

# 2. Dibujar el fondo: Todo el mallado del ACC
if not gdf_mallado.empty:
    gdf_mallado.plot(ax=ax8, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6)

df_sabs_t2.plot(ax=ax8, color='red', edgecolor='black', linewidth=0.5, alpha=0.6)
    
# Crear LEYENDA MANUAL
custom_lines_8 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=0.5, label='SAB del tipo 1'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='SAB del tipo 2')
]

ax8.legend(handles=custom_lines_8, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax8.set_title(f"Resultado del 2º tipo de SAB\nFecha: {fecha_sel} {hora_ini_str}h - {hora_fin_str}h")
ax8.set_xlabel('Longitud [º]')
ax8.set_ylabel('Latitud [º]')
ax8.set_aspect('equal')

plt.tight_layout()
plt.show()


## SAB finales
fig, ax8 = plt.subplots(figsize=(12, 10))

# 1. Dibujar los Sectores del ACC (SIN etiqueta 'label')
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    ax8.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5)
    ax8.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6)

# 2. Dibujar el fondo: Todo el mallado del ACC
if not gdf_mallado.empty:
    gdf_mallado.plot(ax=ax8, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6)

# 3. Dibujar SABs Finales
if not df_sabs_t1_final.empty:
    # Eliminado el figsize interno que era redundante
    df_sabs_t1_final.plot(ax=ax8, color='cyan', alpha=0.5, edgecolor='blue')

df_sabs_t2.plot(ax=ax8, color='red', edgecolor='black', linewidth=0.5, alpha=0.6)
    

# 4. Crear LEYENDA MANUAL
custom_lines_8 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=0.5, label='SAB del tipo 1'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='SAB del tipo 2')
]

ax8.legend(handles=custom_lines_8, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax8.set_title(f"Resultado final del diseño de SAB\nFecha: {fecha_sel} {hora_ini_str}h - {hora_fin_str}h")
ax8.set_xlabel('Longitud [º]')
ax8.set_ylabel('Latitud [º]')
ax8.set_aspect('equal')

plt.tight_layout()
plt.show()







#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- REPORTE Y GUARDADO DE DATOS -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
