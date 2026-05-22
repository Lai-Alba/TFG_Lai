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
import pandas as pd
import geopandas as gpd
import pickle
import shap
import time
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

from datetime import datetime
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from turtle import color



start_time = time.time()

# DIRECTORIOS - ACC Madrid Norte

#! PENDIENTE DE ELIMINAR DIRECTORIOS INNECESARIOS
PATH_TRAFICO = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\Junio2022\\'
PATH_SECTOR_DATA = 'F:\\Users\\Lai\\Datos\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'
PATH_flujos = 'F:\\Users\\Lai\\original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\'
PATH_mallado = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\'
PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\Junio2022\\'
PATH_COMPLEJIDAD = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Junio2022\\test\\' # Asegúrate de que esta ruta es correcta
PATH_resultados = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\Junio2022\\test\\'
PATH_real = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\mallado_20x20\\'

# configuración del estudio (ejemplo: CNF5A, CNF5B, etc.)
#! PENDIENTE DE AUTOMATIZAR LA SELECCIÓN DE CONFIGURACIÓN (ej. con input o argumentos)
configuracion_estudio = 'CNF5A'

# Día seleccionado
#fecha_sel = input("Introduce un día del junio de 2022 (2022-06-DD): ")
fecha_sel = 1
fecha_sel = f"{fecha_sel:02d}"  # Formatear con ceros a la izquierda (ej. 01, 02, ..., 30) 

# Crear el objeto de fecha y el nombre de la carpeta (formato YYYY-MM-DD)
fecha_data = datetime(2022, 6, int(fecha_sel))
nombre_carpeta = f"RESULTADOS_{fecha_data:%Y-%m-%d}"

# 4. Unir la ruta base con la nueva carpeta
PATH_COMPLEJIDAD_DIA = os.path.join(PATH_COMPLEJIDAD, nombre_carpeta)

# Tamaño de celda en nm
#! PENDIENTE DE AUTOMATIZAR LA SELECCIÓN DE TAMAÑO DE CELDA
# Es difícil automatizarlo ya que los datasets generados se basan el tamaño de 20
cell_size_nm = 20
# cell_size_nm = input("Introduce el tamaño de celda en NM (ej. 20): ")

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

## Datos de flujos, celdas, sabs obtenidos, sectores y de mallado------------------------------------------------------------------------------- 

# DATASET ANÁLISIS FLUJOS POR CELDA: qué flujos atraviesan cada celda
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda_{fecha_sel}_filtrado.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo_{fecha_sel}_filtrado.pkl')

# Datos de los sabs detectados
sab_tipo1 = pd.read_pickle(PATH_sabs + f'SAB_tipo1_final_2022-06-{fecha_sel}.pkl')
sab_tipo2 = pd.read_pickle(PATH_sabs + f'SAB_tipo2_2022-06-{fecha_sel}.pkl')
forma_sab_1 = pd.read_pickle(PATH_sabs + f'SAB_tipo1_definitivo_2022-06-{fecha_sel}.pkl')

# Datos del mallado de tamaño 20x20 nm
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')

# Datos de los sectores (con su geometría)
DF_info_conf = pd.read_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')

# Datos de los flujos clusterizados
df_espinazos = pd.read_pickle(PATH_sabs + f'espinazos_flujos_{fecha_sel}.pkl')

# Datos de vuelos reales
DT_real = pd.read_pickle(PATH_real + f'DF_T_REAL_CELDA_20x20_{fecha_sel}.pkl')


## Procesado para obtener Geodataframes------------------------------------------------------------------------------- 

gdf_sab2 = gpd.GeoDataFrame(sab_tipo2, geometry='forma de SAB')

gdf_mallado = gpd.GeoDataFrame(df_mallado, geometry='Polygon')
gdf_mallado = gdf_mallado.rename_geometry('geometry')

## Procesado para obtener las fronteras internas de los sectores------------------------------------------------------------------------------- 

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

# 5. Visualización de las fronteras internas
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

## Datos de complejidad----------------------------------------------------------------------------

# 1. Cargar los datos
df_complejidad_sum = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Suma_2022-06-{fecha_sel}_Completo.pkl')
df_complejidad_media = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Media_2022-06-{fecha_sel}_Diaria.pkl')

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



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- NIVELES DE VUELO DE TIPO 1 -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

flujo_1 = sab_tipo1['Flujo'].unique().tolist()
data_real = DT_real[DT_real['Flujo_Clusterizado'].isin(flujo_1)]

print(data_real['Flujo_Clusterizado'].unique())

FL_tipo1 =[]


for f in flujo_1:
    FL_in = set()
    FL_out = set()
    print(f"Procesando Flujo: {f}")

    for _, data in data_real.iterrows():
        print(f"Flujo_Clusterizado: {data['Flujo_Clusterizado']}")

        if data['Flujo_Clusterizado'] == f:
            FL_in.add(data['modoCIN'])
            FL_out.add(data['modoCOUT'])
    
    todos_los_FL = FL_in.union(FL_out)

    if todos_los_FL: # Comprobamos que no esté vacío para evitar errores
        intervalo_FL = (min(todos_los_FL), max(todos_los_FL))
    else:
        intervalo_FL = (None, None)

    FL_tipo1.append({
        'Flujo': f, 
        'FL_in': FL_in, 
        'FL_out': FL_out,
        'Intervalo de FL': intervalo_FL
    })



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- NIVELES DE VUELO DE TIPO 1 -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #



FL_tipo2 =[]
flujo_2 = []
data_real_2 = []

for idx, row in sab_tipo2.iterrows():
    flujo_2 = row['Flujos de SAB']
    
    nodo = row['nodo_id']
    flujo_ids = [item['Original_ID'] for item in flujo_2]
    data_real_2 = DT_real[DT_real['Flujo_Clusterizado'].isin(flujo_ids)]

    
    print(f"Procesando flujos: {flujo_2}")

    for f in flujo_ids:
        FL_in = set()
        FL_out = set()

        for _, data in data_real_2.iterrows():
            if data['Flujo_Clusterizado'] == f:
                FL_in.add(data['modoCIN'])
                FL_out.add(data['modoCOUT'])

        todos_los_FL = list(FL_in.union(FL_out))
        min_Fl = min(todos_los_FL)
        max_Fl = max(todos_los_FL)
        media_FL = np.mean(todos_los_FL)

        FL_tipo2.append({
            'Flujo': f, 
            'nodo_id': nodo,
            'FL_in': FL_in, 
            'FL_out': FL_out,
            'FL minimo': min_Fl,
            'FL maximo': max_Fl,
            'FL medio': media_FL
        })

FL_min_2 = np.mean([item['FL minimo'] for item in FL_tipo2 if item['FL minimo'] is not None])
FL_max_2 = np.mean([item['FL maximo'] for item in FL_tipo2 if item['FL maximo'] is not None])

print(f"FL mínimo promedio en SAB tipo 2: {FL_min_2}")
print(f"FL máximo promedio en SAB tipo 2: {FL_max_2}")

df_fl = pd.DataFrame(FL_tipo2)
df_fl.to_pickle(PATH_sabs + f'FL_tipo2_2022-06-{fecha_sel}.pkl')
df_fl.to_csv(PATH_sabs + f'FL_tipo2_2022-06-{fecha_sel}.csv', index=False)
print("Guardado")

# Limpiar posibles nulos
df_fl = df_fl.dropna(subset=['FL minimo', 'FL maximo', 'FL medio'])

# 2. Definir los intervalos fijos (de 320 a 480, de 10 en 10)
# Ponemos 490 como límite superior en el 'range' para asegurarnos de que 
# el último bloque (ej. de 470 a 480) se dibuje correctamente.
bins = range(100, 660, 10)

# 3. Crear la figura con 3 subgráficos (1 fila, 3 columnas)
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
fig.suptitle('Distribución de Niveles de Vuelo (Agrupados de 10 en 10)', fontsize=16, fontweight='bold')

# --- Gráfico 1: FL Mínimo ---
axes[0].hist(df_fl['FL minimo'], bins=bins, orientation='horizontal', color='skyblue', edgecolor='black', alpha=0.8)
axes[0].set_title('FL Mínimo')
axes[0].set_xlabel('Número de Flujos')
axes[0].set_ylabel('Niveles de Vuelo (FL)')

# --- Gráfico 2: FL Máximo ---
axes[1].hist(df_fl['FL maximo'], bins=bins, orientation='horizontal', color='salmon', edgecolor='black', alpha=0.8)
axes[1].set_title('FL Máximo')
axes[1].set_xlabel('Número de Flujos')

# --- Gráfico 3: FL Medio ---
axes[2].hist(df_fl['FL medio'], bins=bins, orientation='horizontal', color='lightgreen', edgecolor='black', alpha=0.8)
axes[2].set_title('FL Medio')
axes[2].set_xlabel('Número de Flujos')

# 4. Ajustes visuales para los 3 gráficos
for ax in axes:
    ax.grid(axis='x', linestyle='--', alpha=0.7) # Cuadrícula para ver bien la cantidad de flujos
    ax.set_yticks(bins) # Forzar marcas en el eje Y exactas (320, 330, 340...)
    ax.set_ylim(100, 660) # Fijar estrictamente el eje Y visual entre 320 y 480

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%
# ==================================================================================================================== #
# ======================================= 4. VISUALIZACIÓN 3D CON TRIMESH ============================================ #
# ==================================================================================================================== #
import trimesh
from shapely.geometry import Polygon, MultiPolygon

print("Generando modelo 3D...")

# 1. FACTOR DE ESCALA PARA Z
# Las coordenadas XY están en grados (lat/lon). 1 grado ~ 111 km.
# El FL320 son 32000 pies (~9.7 km). 
# Aplicamos un factor de escala para que la visualización 3D tenga proporciones lógicas.
escala_z = 0.002 # Ajusta este valor si ves los bloques muy altos o muy planos

# Lista para guardar todas las mallas de la escena
objetos_escena = []

# -------------------------------------------------------------------------
# A. CREACIÓN DE LA PLANTILLA BASE: ACC MADRID NORTE EN Z = 0
# -------------------------------------------------------------------------
# Obtenemos la unión de todos los sectores para hacer la "huella" del ACC
acc_polygon = gdf_sectores.union_all()
color_plantilla = [0, 120, 255, 40] # Azul muy traslúcido

# Función auxiliar para triangular polígonos simples o múltiples
def añadir_plano_2d_a_3d(geometria, color, z_level=0.0):
    poligonos = geometria.geoms if isinstance(geometria, MultiPolygon) else [geometria]
    for p in poligonos:
        try:
            vertices_2d, faces = trimesh.creation.triangulate_polygon(p)
            vertices_3d = np.hstack((vertices_2d, np.full((len(vertices_2d), 1), z_level)))
            mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
            mesh.visual.face_colors = color
            objetos_escena.append(mesh)
        except Exception as e:
            print(f"Polígono base omitido por error de triangulación: {e}")

añadir_plano_2d_a_3d(acc_polygon, color_plantilla, z_level=0.0)

# -------------------------------------------------------------------------
# B. CREACIÓN DE BLOQUES 3D: SAB TIPO 1
# -------------------------------------------------------------------------
# Iteramos sobre la lista FL_tipo1 que creaste antes
for item in FL_tipo1:
    flujo = item['Flujo']
    min_fl, max_fl = item['Intervalo de FL']
    
    if min_fl is not None and max_fl is not None:
        # Extraemos la geometría asociada a este flujo en sab_tipo1
        # (Asegúrate de que la columna de geometría en sab_tipo1 se llama 'forma de SAB' o ajusta el nombre)
        fila_sab1 = forma_sab_1[forma_sab_1['Flujo_ID'] == flujo]
        
        if not fila_sab1.empty:
            geometria = fila_sab1.iloc[0]['geometry'] # Cambiar si la columna se llama distinto
            
            espesor = (max_fl+20 - min_fl+20) * escala_z
            if espesor == 0: espesor = 0.1 * escala_z # Dar un espesor mínimo si min_fl == max_fl
            base_z = min_fl * escala_z
            
            print(f"intervalo = {max_fl} - {min_fl} = {max_fl - min_fl}")
            print(f"espesor = {espesor}")

            poligonos = geometria.geoms if isinstance(geometria, MultiPolygon) else [geometria]
            for p in poligonos:
                try:
                    modelo_sab = trimesh.creation.extrude_polygon(p, height=espesor)
                    modelo_sab.apply_translation([0.0, 0.0, base_z])
                    modelo_sab.visual.face_colors = [255, 100, 100, 180] # Rojo semitransparente
                    objetos_escena.append(modelo_sab)
                except:
                    pass

# -------------------------------------------------------------------------
# C. CREACIÓN DE BLOQUES 3D: SAB TIPO 2
# -------------------------------------------------------------------------
# # Iteramos sobre el DataFrame df_fl que creaste y guardaste
# for idx, row in df_fl.iterrows():
#     nodo = row['nodo_id']
#     min_fl = row['FL minimo']
#     max_fl = row['FL maximo']
    
#     # Buscamos la geometría asociada en sab_tipo2
#     fila_sab2 = sab_tipo2[sab_tipo2['nodo_id'] == nodo]
    
#     if not fila_sab2.empty:
#         geometria = fila_sab2.iloc[0]['forma de SAB']
        
#         espesor_l = (345-0) * escala_z
#         espesor_u = (660-345) * escala_z

#         # if espesor == 0: espesor = 0.1 * escala_z # Dar un espesor mínimo
#         base_z_u = 345* escala_z
#         base_z = 0
#         poligonos = geometria.geoms if isinstance(geometria, MultiPolygon) else [geometria]
#         for p in poligonos:
#             try:
#                 modelo_sab2_1 = trimesh.creation.extrude_polygon(p, height=espesor_l)
#                 modelo_sab2_1.apply_translation([0.0, 0.0, base_z])
#                 modelo_sab2_1.visual.face_colors = [100, 255, 100, 180] # Verde semitransparente
#                 objetos_escena.append(modelo_sab2_1)

#                 modelo_sab2_2 = trimesh.creation.extrude_polygon(p, height=espesor_u)
#                 modelo_sab2_2.apply_translation([0.0, 0.0, base_z_u])
#                 modelo_sab2_2.visual.face_colors = [255, 100, 100, 180] # Verde semitransparente
#                 objetos_escena.append(modelo_sab2_2)
#             except:
#                 pass

# -------------------------------------------------------------------------
# D. EJES DE REFERENCIA Y RENDERIZADO
# -------------------------------------------------------------------------
# En lugar de poner los ejes en (0,0,0), los ponemos cerca del centro del ACC para verlos mejor
centro_acc = acc_polygon.centroid
ejes = trimesh.creation.axis(origin_size=0.1, axis_length=2.0)
# Desplazamos los ejes al centroide del mapa
ejes.apply_translation([centro_acc.x, centro_acc.y, 0])
objetos_escena.append(ejes)

# Creamos y mostramos la escena
escena = trimesh.Scene(objetos_escena)
print("Mostrando modelo 3D de los SABs...")
escena.show()
# escena.export(PATH_sabs + f"modelo_final_{fecha_sel}-06.glb")















# %%
