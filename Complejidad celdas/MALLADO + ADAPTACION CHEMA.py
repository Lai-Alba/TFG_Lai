

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import itertools
import warnings
import re
import os
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import box, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import itertools
from itertools import product
from matplotlib.lines import Line2D
import geopy.distance
from geopy.distance import geodesic
from shapely import wkt
from shapely.wkt import loads
import pickle
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import nearest_points
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from datetime import datetime
import shap
import time
import seaborn as sns
import gc
import ast
import math
from shapely.prepared import prep
from shapely.strtree import STRtree



import time
start_time = time.time()
# DIRECTORIOS - ACC Madrid Norte

PATH_TRAFICO = "F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\"
PATH_SECTOR_DATA = "F:\\Users\\Lai\\Datos\\1. bloque prediccion\\datos\\ACC Madrid\\Sector Data\\LECMCTAN\\"
PATH_flujos = "F:\\Users\\Lai\\Original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\"
PATH_resultados = "F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\"
PATH_TRAFICO_CELDA = "F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\"

configuracion_estudio = 'CNF9A2'




#%%
########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------- PRIMERA PARTE DEL CÓDIGO: CARACTERIZACIÓN DEL ESPACIO AÉREO A NIVEL CELDA --------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ IMPORTACIÓN DE BASES DE DATOS ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# IMPORTACIÓN DE LA BASE DE DATOS DE FLUJOS CLUSTERIZADOS
DF_Flujos_ = pd.read_csv(
    PATH_flujos + "flow_trend_DF.csv",
    sep=";",
    encoding="latin1",
    dtype=None,        # intenta inferir tipos
    parse_dates=True,  # intenta convertir fechas
    low_memory=False
)
DF_Flujos = DF_Flujos_.copy()
# DF_Flujos_ = pd.read_pickle(PATH_flujos + 'flow_trend_DF.pkl')
# DF_Flujos = DF_Flujos_.copy() # Como es una porción de otro DataFrame, usar .copy() para evitar un futuro warning con pandas

# Crear objetos geométricos para los flujos -> línea recta que representa el flujo
DF_Flujos.loc[:, 'Line'] = DF_Flujos.apply(lambda row: LineString([(row['lon_f_in'], row['lat_f_in']), (row['lon_f_out'], row['lat_f_out'])]), axis=1)



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------- CREACION DE LOS DATOS NECESARIOS PARA GRAFICAR ---------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# LECTUTRA DE LAS CONFIGURACIONES DEL ACC SELECCIONADO
config = pd.read_csv(PATH_SECTOR_DATA + 'config.txt',sep='\t', header=None)
config = config[0].str.split(';', expand=True)
ACC = config[0].iloc[0]
print('El ACC de la base de datos es', ACC)
config = config.rename(columns={1: 'CONFIG', 2: 'SECTORES'})

list_dataframes = [df for df in config.groupby('CONFIG', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    configuracion = df['CONFIG'].iloc[0]
    sectors = list(df['SECTORES'])
    df_temporal = pd.DataFrame({'CONFIG': [configuracion], 'SECTORES': [sectors]})
    dataframes_temporales.append(df_temporal)
CONFIG = pd.concat(dataframes_temporales, ignore_index=True)
del (config)


# LECTURA DE LA COMPOSICION DE LOS ESPACIOS AEREOS
airspaces = pd.read_csv(PATH_SECTOR_DATA + 'airspace.txt',sep='\t', header=None)
airspaces = airspaces.drop(airspaces.index[0])
airspaces = airspaces[0].str.split(';', expand=True)

rows = []
current_id = None
current_nombre = None
for index, row in airspaces.iterrows():
    if 'A' in row.iloc[0]:
        current_id = row.iloc[1]
        current_nombre = row.iloc[2]
        tipo = row.iloc[3]
        number = row.iloc[4]
        rows.append((current_id, current_nombre, tipo, number))
    else:
        rows.append((current_id, current_nombre, tipo, number, row.iloc[1]))

AIRSPACES = pd.DataFrame(rows)
AIRSPACES = AIRSPACES.rename(columns={0: 'AIRSPACE_ID', 1: 'NOMBRE', 2: 'TIPO', 3: 'NUMBER', 4: 'SECTORES'})
AIRSPACES = AIRSPACES.dropna(subset=['SECTORES'])
AIRSPACES = AIRSPACES.reset_index(drop=True)


list_dataframes = [df for df in AIRSPACES.groupby('AIRSPACE_ID', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    id_airspace = df['AIRSPACE_ID'].iloc[0]
    nombre = df['NOMBRE'].iloc[0]
    bloques = list(df['SECTORES'])
    tipo = df['TIPO'].iloc[0]
    number = df['NUMBER'].iloc[0]
    df_temporal = pd.DataFrame({'AIRSPACE_ID': [id_airspace], 'NOMBRE': [nombre], 'TIPO': [tipo], 'NUMBER': [number], 'SECTORES': [bloques]})
    dataframes_temporales.append(df_temporal)

AIRSPACES = pd.concat(dataframes_temporales, ignore_index=True)
del (airspaces)


# LECTURA DE LA COMPOSICION DE LOS SECTORES
sectores = pd.read_csv(PATH_SECTOR_DATA + 'sectors.txt',sep='\t', header=None)
sectores = sectores.drop(sectores.index[0])
sectores = sectores[0].str.split(';', expand=True)

rows = []
current_id = None
current_nombre = None
for index, row in sectores.iterrows():
    if 'S' in row.iloc[0]:
        current_id = row.iloc[1]
        current_nombre = row.iloc[2]
        rows.append((current_id, current_nombre,))
    else:
        rows.append((current_id, current_nombre, row.iloc[1], row.iloc[4]))
SECTORES = pd.DataFrame(rows)
SECTORES = SECTORES.rename(columns={0: 'SECTOR_ID', 1: 'NOMBRE', 2: 'AIR BLOCKS', 3: 'MAX FL'})
SECTORES = SECTORES.dropna(subset=['AIR BLOCKS'])
SECTORES = SECTORES.reset_index(drop=True)

list_dataframes = [df for df in SECTORES.groupby('SECTOR_ID', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    # print(df)
    id_sector = df['SECTOR_ID'].iloc[0]
    nombre = df['NOMBRE'].iloc[0]
    bloques = list(df['AIR BLOCKS'])
    max_FL = list(df['MAX FL'])
    # print(bloques)
    df_temporal = pd.DataFrame({'SECTOR_ID': [id_sector], 'NOMBRE': [nombre], 'AIR BLOCKS': [bloques], 'MAX FL': [max_FL]})
    dataframes_temporales.append(df_temporal)
SECTORES = pd.concat(dataframes_temporales, ignore_index=True)


# LECTURA DE LOS BLOQUES DE ESPACIO AEREO
bloques = pd.read_csv(PATH_SECTOR_DATA + 'bloques.txt',sep='\t', header=None)
bloques = bloques[0].str.split(';', expand=True)
bloques = bloques.drop(bloques.index[0])

rows = []
current_id = None
for index, row in bloques.iterrows():
    if 'A' in row.iloc[0]:
        current_id = row.iloc[1]
        rows.append((current_id, current_id))
    else:
        rows.append((current_id, row.iloc[1], row.iloc[2]))
bloques = pd.DataFrame(rows)
bloques = bloques.rename(columns={0: 'ID_BLOQUE', 1: 'LAT', 2: 'LON'})
bloques = bloques.dropna(subset=['LON'])
bloques = bloques.reset_index(drop=True)

list_dataframes = [df for df in bloques.groupby('ID_BLOQUE', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    id_vuelo = df['ID_BLOQUE'].iloc[0]
    coordenadas = list(zip(df['LAT'], df['LON']))
    df_temporal = pd.DataFrame({'ID_BLOQUE': [id_vuelo], 'Coordenadas': [coordenadas], })
    dataframes_temporales.append(df_temporal)
BLOQUES = pd.concat(dataframes_temporales, ignore_index=True)


# CREAR POLIGONOS CON LOS BLOQUES DE ESPACIO AEREO
BLOQUES['Contorno Bloque'] = None
for index, row in BLOQUES.iterrows():
    coordenadas = row['Coordenadas']
    y_coords = [coord[0] for coord in coordenadas]
    x_coords = [coord[1] for coord in coordenadas]
    poligono = Polygon(zip(x_coords, y_coords))
    BLOQUES.loc[index, 'Contorno Bloque'] = poligono


# CREAR SECTORES ELEMENTALES
SECTORES['Contorno Sector'] = None
SECTORES['TIPO'] = 'EL'
SECTORES['ACC'] = ACC
for index, row in SECTORES.iterrows():
    bloques = row['AIR BLOCKS']
    for bloque in bloques:
        poligono = BLOQUES.loc[BLOQUES['ID_BLOQUE'] == bloque, 'Contorno Bloque'].values[0]
        if row['Contorno Sector'] is None:
            row['Contorno Sector'] = poligono
        else:
            row['Contorno Sector'] = row['Contorno Sector'].union(poligono)
    SECTORES.loc[index, 'Contorno Sector'] = row['Contorno Sector']

SECTORES2 = pd.concat([SECTORES['SECTOR_ID'], SECTORES['Contorno Sector'], SECTORES['TIPO'],
                       SECTORES['ACC']], axis=1)


# CREAR SECTORES COLAPSADOS
AIRSPACES['ACC'] = ACC
AIRSPACES['Contorno Sector Colapsado'] = None
for index, row in AIRSPACES.iterrows():
    sectors = row['SECTORES']
    for sector in sectors:
        # print('Sector:', sector)
        poligono = SECTORES.loc[SECTORES['SECTOR_ID'] == sector, 'Contorno Sector'].values[0]
        if row['Contorno Sector Colapsado'] is None:
            row['Contorno Sector Colapsado'] = poligono
        else:
            row['Contorno Sector Colapsado'] = row['Contorno Sector Colapsado'].union(poligono)
    AIRSPACES.loc[index, 'Contorno Sector Colapsado'] = row['Contorno Sector Colapsado']

AIRSPACES2 = pd.concat([AIRSPACES['AIRSPACE_ID'], AIRSPACES['Contorno Sector Colapsado'], AIRSPACES['TIPO'],
                        AIRSPACES['ACC']], axis=1)
# RENOMBRA COLUMNAS
AIRSPACES2 = AIRSPACES2.rename(columns={'AIRSPACE_ID': 'SECTOR_ID', 'Contorno Sector Colapsado': 'Contorno Sector'})



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------- REPRESENTACIÓN DE LOS SECTORES DEL ACC -------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

print('Configuración de estudio seleccionada:',configuracion_estudio)

# SECTORES DE LA CONFIGURACIÓN
list_sectors = CONFIG.loc[CONFIG['CONFIG'] == configuracion_estudio, 'SECTORES'].iloc[0]
print('Sectores de la configuración:', list_sectors)

# Filtrar datos según los sectores seleccionados
resultado1 = AIRSPACES[AIRSPACES['AIRSPACE_ID'].isin(list_sectors)]
resultado1 = resultado1.rename(columns={'AIRSPACE_ID': 'SECTOR_ID', 'Contorno Sector Colapsado': 'Contorno Sector'})
resultado2 = SECTORES[SECTORES['SECTOR_ID'].isin(list_sectors)]

# Combinar resultados en un solo DataFrame
DF_info_conf = pd.concat([resultado1, resultado2]).reset_index(drop=True)

#OBTENER LA MAXIMA LATITUD Y LONGITUD DEL ACC
min_lat = []
max_lat = []
min_lon = []
max_lon = []
for index, row in DF_info_conf.iterrows():
    poligono = row['Contorno Sector']
    x, y = poligono.exterior.xy
    min_lat.append(min(y))
    max_lat.append(max(y))
    min_lon.append(min(x))
    max_lon.append(max(x))

min_lat = min(min_lat) -0.5
max_lat = max(max_lat) +0.5
min_lon = min(min_lon) -0.5
max_lon = max(max_lon) +0.5


#PLOTEAR EL ACC
fig, sects = plt.subplots()
sects.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
sects.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
sects.set_aspect('equal')
plt.xlabel('LONGITUD[º]')
plt.ylabel('LATITUD[º]')
plt.title('REPRESENTACION DE LOS SECTORES DE ESTUDIO')

for index, row in DF_info_conf.iterrows():
    poligono = row['Contorno Sector']
    x, y = poligono.exterior.xy
    sects.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'{row["SECTOR_ID"]}')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- OBTENCIÓN DEL MALLADO DEL ACC ------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DEL ESPACIO AÉREO ASOCIADO AL ACC
# Unir todos los polígonos de los sectores en un único polígono o MultiPolygon
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

# Obtener el contorno exterior del conjunto de sectores
if isinstance(union_poligonos, MultiPolygon):
    poligono_ACC = union_poligonos.convex_hull
else:
    poligono_ACC = union_poligonos.exterior

# Convertir LinearRing a Polygon
poligono_ACC = Polygon(poligono_ACC)


# REPRESENTACIÓN DEL ESPACIO AÉREO ASOCIADO AL ACC
min_lat = []
max_lat = []
min_lon = []
max_lon = []

x, y = poligono_ACC.exterior.xy
min_lat.append(min(y))
max_lat.append(max(y))
min_lon.append(min(x))
max_lon.append(max(x))

min_lat = min(min_lat) -0.5
max_lat = max(max_lat) +0.5
min_lon = min(min_lon) -0.5
max_lon = max(max_lon) +0.5

fig, ACC = plt.subplots()
ACC.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ACC.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ACC.set_aspect('equal')
plt.xlabel('LONGITUD[º]')
plt.ylabel('LATITUD[º]')
plt.title('REPRESENTACION DEL ESPACIO AÉREO DE ESTUDIO - ACC MADRID NORTE')
ACC.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'LECMCTAN')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()


# MALLADO DEL ACC
# Tamaño de celda en millas náuticas
cell_size_nm = 25

# Convertir tamaño de celdas de NM a grados de latitud (constante)
cell_size_lat_deg = cell_size_nm / 60.0

# Obtener los límites del sector
minx, miny, maxx, maxy = poligono_ACC.bounds

# Crear celdas ajustando longitud a partir de latitud promedio del sector
lat_center = (miny + maxy) / 2
cell_size_lon_deg = cell_size_nm / (math.cos(math.radians(lat_center)) * 60)

# Lista para almacenar los datos de las celdas
cell_data = []

# Crear la malla desde la esquina superior izquierda (con soporte para MultiPolygon)
cells = []
cell_id = 1  # Iniciar el contador para las celdas
current_lat = maxy  # Iniciar desde la latitud máxima (norte)

while current_lat > miny:  # Decrecer latitud hacia el sur
    current_lon = minx  # Iniciar desde la longitud mínima (oeste)
    while current_lon < maxx:  # Aumentar longitud hacia el este
        # Crear la celda
        cell = box(current_lon, current_lat - cell_size_lat_deg, current_lon + cell_size_lon_deg, current_lat)
        # Recortar la celda con el espacio aéreo
        intersected_cell = cell.intersection(poligono_ACC)

        if not intersected_cell.is_empty:
            if isinstance(intersected_cell, Polygon):
                coords = list(intersected_cell.exterior.coords)
                cells.append(intersected_cell)
                cell_data.append({
                    'Cell_Name': f'Cell_{cell_id}',
                    'Polygon': intersected_cell,
                    'Coordinates': coords
                })
                cell_id += 1

            elif isinstance(intersected_cell, MultiPolygon):
                for poly in intersected_cell.geoms:
                    coords = list(poly.exterior.coords)
                    cells.append(poly)
                    cell_data.append({
                        'Cell_Name': f'Cell_{cell_id}',
                        'Polygon': poly,
                        'Coordinates': coords
                    })
                    cell_id += 1

        current_lon += cell_size_lon_deg  # Moverse hacia el este
    current_lat -= cell_size_lat_deg  # Moverse hacia el sur


# Crear un DataFrame con los datos
DF_cells = pd.DataFrame(cell_data)

# Graficar el sector con las celdas recortadas
fig, ax_cells = plt.subplots()
ax_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_cells.set_aspect('equal')

# Dibujar el polígono del sector
x, y = poligono_ACC.exterior.xy
ax_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

# Dibujar las celdas recortadas
for cell in cells:
    if isinstance(cell, Polygon):
        x, y = cell.exterior.xy
        ax_cells.plot(x, y, color='gray', alpha=0.5)
    elif isinstance(cell, MultiPolygon):
        for poly in cell.geoms:
            x, y = poly.exterior.xy
            ax_cells.plot(x, y, color='gray', alpha=0.5)

# Configurar la gráfica
ax_cells.set_title("MALLADO DEL ESPACIO AÉREO CON CELDAS 25NM x 25NM")
ax_cells.set_aspect('equal')
ax_cells.set_xlabel('LONGITUD[º]')
ax_cells.set_ylabel('LATITUD[º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- ANÁLISIS CELDAS POR FLUJO -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DE LAS CELDAS DEL MALLADO POR LAS QUE PASA CADA FLUJO - ordenando las celdas según el sentido del flujo

# def puntos_flujo(line_coords, num_puntos=1000):
#     """ Genera puntos equidistantes a lo largo de la LineString. """
#     start_point = line_coords[0]
#     end_point = line_coords[-1]
#     distancia_total = Point(start_point).distance(Point(end_point))
#     puntos = [start_point]

#     for i in range(1, num_puntos):
#         factor = i / num_puntos
#         punto_intermedio = (
#             start_point[0] + (end_point[0] - start_point[0]) * factor,
#             start_point[1] + (end_point[1] - start_point[1]) * factor
#         )
#         puntos.append(punto_intermedio)

#     puntos.append(end_point)  # Añadir el punto final
#     return puntos


# def celdas_por_flujo(flujo, celdas, num_puntos=1000):
#     punto_origen = flujo['Line'].coords[0]  # Punto de origen: primer punto de la línea
#     line_coords = list(flujo['Line'].coords)

#     # Generar puntos equidistantes a lo largo de la línea del flujo
#     puntos_intermedios = puntos_flujo(line_coords, num_puntos)

#     celdas_visitadas = []
#     for coord in puntos_intermedios:
#         punto_actual = Point(coord)

#         celdas_intersectadas = [celda['Cell_Name'] for _, celda in celdas.iterrows()
#                                 if celda['Polygon'].contains(punto_actual)]

#         celdas_visitadas.extend(celdas_intersectadas)

#     # Eliminar celdas duplicadas y mantener el orden
#     celdas_visitadas = list(dict.fromkeys(celdas_visitadas))  # Eliminar duplicados conservando el orden
#     return celdas_visitadas


# # Aplicar la función a DF_Flujos_sector
# DF_Flujos.loc[:, 'Cell_Names'] = DF_Flujos.apply(lambda flujo: celdas_por_flujo(flujo, DF_cells), axis=1)

# # Imprimir el resultado
# print(DF_Flujos[['Flujo_Clusterizado', 'Cell_Names']])

# # GUARDAR LAS BASES DE DATOS DE LAS CELDAS POR LAS QUE PASA CADA FLUJO
# DF_Flujos.to_csv(PATH_resultados + 'dataset_celdas_por_flujo.csv', index=False, sep=';')
# DF_Flujos.to_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')

# # Me tarda la Vida en correr esta sección, hay que buscar una forma de que no tenga que borrar los puntos para celdas repetidas

# ---------------------------
# MODO PRUEBAS (reversible)
# ---------------------------
FAST_MODE = True
N_FLOWS_PRUEBA = 200          # procesa solo los primeros N flujos (debug)
NUM_PUNTOS_FAST = 150         # en vez de 1000
GUARDAR_EN_DISCO = False      # no guardes mientras iteras rápido

def puntos_flujo(line_coords, num_puntos=100):
    start_point = line_coords[0]
    end_point = line_coords[-1]
    puntos = [start_point]
    for i in range(1, num_puntos):
        factor = i / num_puntos
        puntos.append((
            start_point[0] + (end_point[0] - start_point[0]) * factor,
            start_point[1] + (end_point[1] - start_point[1]) * factor
        ))
    puntos.append(end_point)
    return puntos

# ---------------------------
# PRECOMPUTOS (CLAVE)
# ---------------------------
cell_names = DF_cells['Cell_Name'].to_list()
cell_polys = DF_cells['Polygon'].to_list()

# Prepared geometries: aceleran contains/covers
cell_prepared = [prep(p) for p in cell_polys]

# STRtree: índice espacial para no recorrer todas las celdas
tree = STRtree(cell_polys)

# Compatibilidad Shapely 1.x (query devuelve geometrías) / Shapely 2.x (puede devolver índices)
geom_to_idx = {id(g): i for i, g in enumerate(cell_polys)}

def _candidate_indices_from_query(qres):
    if len(qres) == 0:
        return []
    # Shapely 2: devuelve índices (numpy ints)
    if isinstance(qres[0], (int, np.integer)):
        return list(map(int, qres))
    # Shapely 1: devuelve geometrías
    return [geom_to_idx.get(id(g), None) for g in qres if id(g) in geom_to_idx]

def celdas_por_flujo_fast(flujo, num_puntos=NUM_PUNTOS_FAST):
    line = flujo['Line']
    coords = list(line.coords)
    pts = puntos_flujo(coords, num_puntos)

    visitadas = []
    last_cell = None

    for coord in pts:
        p = Point(coord)

        cand = tree.query(p)  # candidatos cercanos
        idxs = _candidate_indices_from_query(cand)

        found = None
        for idx in idxs:
            if idx is None:
                continue
            # covers incluye borde (a veces mejor que contains si el punto cae justo en frontera)
            if cell_prepared[idx].covers(p):
                found = cell_names[idx]
                break

        # Evitar duplicados "en racha" sin dict.fromkeys
        if found is not None and found != last_cell:
            visitadas.append(found)
            last_cell = found

    return visitadas

# ---------------------------
# APLICACIÓN (conmutador)
# ---------------------------
DF_Flujos_run = DF_Flujos.head(N_FLOWS_PRUEBA).copy() if FAST_MODE else DF_Flujos

if FAST_MODE:
    DF_Flujos_run.loc[:, 'Cell_Names'] = DF_Flujos_run.apply(celdas_por_flujo_fast, axis=1)
else:
    # Tu versión original (lenta): si quieres mantenerla
    DF_Flujos_run.loc[:, 'Cell_Names'] = DF_Flujos_run.apply(lambda flujo: celdas_por_flujo(flujo, DF_cells), axis=1)

print(DF_Flujos_run[['Flujo_Clusterizado', 'Cell_Names']].head(10))

if GUARDAR_EN_DISCO:
    DF_Flujos_run.to_csv(PATH_resultados + 'dataset_celdas_por_flujo.csv', index=False, sep=';')
    DF_Flujos_run.to_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- ANÁLISIS FLUJOS POR CELDA -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DE LOS FLUJOS QUE PASAN POR CADA UNA DE LAS CELDAS DEL MALLADO

# Función para verificar flujos que pasan por la celda
def flujos_por_celda(celda, flujos):
    return [flujo['Flujo_Clusterizado'] for _, flujo in flujos.iterrows() if celda['Polygon'].intersects(flujo['Line'])]

# Aplicar la función a cada celda
DF_cells.loc[:, 'Flujos_Clusterizados'] = DF_cells.apply(lambda celda: flujos_por_celda(celda, DF_Flujos), axis=1)

# Resultado: lista de flujos que pasan por cada celda
print(DF_cells[['Cell_Name', 'Flujos_Clusterizados']])

# GUARDAR LAS BASES DE DATOS DE LOS FLUJOS QUE PASAN POR CADA CELDA
DF_cells.to_csv(PATH_resultados + 'dataset_flujos_por_celda.csv', index=False, sep=';')
DF_cells.to_pickle(PATH_resultados + 'dataset_flujos_por_celda.pkl')



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- COMPROBACIONES GRÁFICAS --------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# COMPROBACION GRÁFICA DE TODAS LAS CELDAS POR LAS QUE PASA CADA FLUJO

# Elegir un flujo cualquiera
flujo_elegido = DF_Flujos.iloc[242]  # Cambiar el índice aquí para elegir otro flujo

# Crear una figura y un eje para la gráfica
fig, ax_2 = plt.subplots()

# Graficar las celdas del mallado (polígonos)
for _, celda in DF_cells.iterrows():
    x, y = celda['Polygon'].exterior.xy
    ax_2.fill(x, y, alpha=0.3, color='lightblue', edgecolor='blue')
    # Colocar nombre de la celda en el centro del polígono
    ax_2.text(celda['Polygon'].centroid.x, celda['Polygon'].centroid.y, celda['Cell_Name'],fontsize=4, ha='center', color='black')

# Graficar el flujo elegido (línea)
x_flujo, y_flujo = flujo_elegido['Line'].xy
ax_2.plot(x_flujo, y_flujo, color='red', linewidth=2)

# Colocar el nombre del flujo
ax_2.text(flujo_elegido['Line'].centroid.x, flujo_elegido['Line'].centroid.y, flujo_elegido['Flujo_Clusterizado'],fontsize=10, ha='center', color='red')

# Personalizar el gráfico
ax_2.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_2.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_2.set_title('REPRESENTACIÓN DEL FLUJO SOBRE EL MALLADO')
ax_2.set_xlabel('LONGITUD [º]')
ax_2.set_ylabel('LATITUD [º]')
ax_2.set_aspect('equal')
plt.show()



#%%
########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------- SEGUNDA PARTE DEL CÓDIGO: ADAPTACIÓN DE LA TRAYECTORIA A NIVEL CELDA ------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# # DATASET ANÁLISIS FLUJOS POR CELDA
# DF_cells = pd.read_pickle(PATH_resultados + 'dataset_flujos_por_celda.pkl')

# # DATASET ANÁLISIS CELDAS POR FLUJO
# DF_Flujos = pd.read_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')

# # # SELECCIÓN DEL TRÁFICO DE ESTUDIO
# # tipo_trafico = input('Selecciona el tipo de tráfico a extrapolar al mallado (real/predicciones): ')

# # IMPORTACIÓN DE LA BASE DE DATOS CORRESPONDIENTE AL TIPO DE TRÁFICO SELECCIONADO
# # if tipo_trafico == 'real':
# #     # Importación de la base de datos de tráfico real

# DF_Trafico = pd.read_pickle(PATH_TRAFICO + 'dataset_vuelos_reales_2022-06-01.pkl')

# # elif tipo_trafico == 'predicciones':
# #     # Importación de la base de datos de tráfico basado en predicciones
# #     DF_Trafico = pd.read_pickle(PATH_TRAFICO + 'dataset_vuelos_predicciones_2022-06-01.pkl')

#----------------------------------------------------------------------------------------------------------------

# DATASET ANÁLISIS FLUJOS POR CELDA
DF_cells = pd.read_csv(
    PATH_resultados + "dataset_flujos_por_celda.csv",
    sep=";",
    encoding="latin1",
    dtype=None,        # intenta inferir tipos
    parse_dates=True,  # intenta convertir fechas
    low_memory=False
)

# DATASET ANÁLISIS CELDAS POR FLUJO
DF_Flujos = pd.read_csv(
    PATH_resultados + "dataset_celdas_por_flujo.csv",
    sep=";",
    encoding="latin1",
    dtype=None,
    parse_dates=True,
    low_memory=False
)

# IMPORTACIÓN DE LA BASE DE DATOS CORRESPONDIENTE AL TIPO DE TRÁFICO SELECCIONADO
DF_Trafico = pd.read_csv(
    PATH_TRAFICO + "dataset_vuelos_reales_2022-06-01.csv",
    sep=";",
    encoding="latin1",
    dtype=None,
    parse_dates=True,
    low_memory=False
)

# Si usas predicciones, sería:
# DF_Trafico = pd.read_csv(
#     PATH_TRAFICO + "dataset_vuelos_predicciones_2022-06-01.csv",
#     sep=";",
#     encoding="latin1",
#     dtype=None,
#     parse_dates=True,
#     low_memory=False
# )


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------- INCORPORACIÓN DE LA INFORMACIÓN DEL MALLADO A LAS PREDICCIONES -------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Filtrar las columnas de interés
DF_Trafico = DF_Trafico[['flightKey', 'origen_destino', 'airline', 'aircraftType', 'wake', 'routeType', 'ETOT', 'IOBT', 'Secuencia', 'Sector',
                                   'Flujo_Clusterizado','Clave_Flujo','Flow_Trend','sectorEntryInstant','sectorExitInstant','modoCIN','modoCOUT',
                                   'attitudIN','attitudOUT','t-hasta_sector','t-en_sector','t-salida_sector','Secuencia_Sectores','Secuencia_Flujos',
                                   'Secuencia_Claves_Flujos','Secuencia_Flow_Trend']]




# Unir dataframes manteniendo las filas del DF de la izquierda
DF_TRAFICO = pd.merge(DF_Trafico, DF_Flujos[['Clave_Flujo', 'Sector', 'Flujo_Clusterizado', 'Trend_entrada', 'Trend_salida', 'Cell_Names']],
                      how='left', on=['Clave_Flujo','Sector','Flujo_Clusterizado'])

# Reordenar columnas
columnas = list(DF_TRAFICO.columns) # Obtener el nombre de todas las columnas
columnas.insert(12, columnas.pop(columnas.index('Cell_Names'))) # Colocar 'Cell_Names' en la posición deseada de la lista de columnas
columnas.insert(14, columnas.pop(columnas.index('Trend_entrada'))) # Colocar 'Cell_Names' en la posición deseada de la lista de columnas
columnas.insert(15, columnas.pop(columnas.index('Trend_salida'))) # Colocar 'Cell_Names' en la posición deseada de la lista de columnas
DF_TRAFICO = DF_TRAFICO[columnas] # Reordenar el DataFrame con las columnas en el orden deseado



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------- TENDENCIAS DE ENTRADA Y SALIDA A NIVEL CELDA ----------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DE LA ACTITUD DE ENTRADA Y SALIDA DE LAS CELDAS ASOCIADAS A CADA FLUJO
DF_Flujos_completo = DF_Flujos.copy()

# Función para obtener la tendencia de entrada y de salida de las celdas de paso de cada flujo
def calcular_tendencia_celdas(row):

    # Inicializar las listas de tendencias de entrada y salida
    num_celdas = len(row['Cell_Names'])
    trend_cell_entrada = ['CRUISE'] * num_celdas
    trend_cell_salida = ['CRUISE'] * num_celdas

    flow_trend = row['Flow_Trend']
    trend_entrada = row['Trend_entrada']
    trend_salida = row['Trend_salida']
    flow = row['Flujo_Clusterizado']

    if flow_trend == 'CRUISE':

        if trend_salida == 'CRUISE' and trend_salida == 'CRUISE':
            trend_cell_entrada = trend_cell_entrada
            trend_cell_salida = trend_cell_salida

        elif trend_entrada in ['CLIMB', 'DESCEND'] and trend_salida == 'CRUISE':
            # Todas las celdas siguen la trend_salida a excepción de la entrada a la primera celda
            trend_cell_entrada[0] = trend_entrada

        elif trend_entrada == 'CRUISE' and trend_salida in ['CLIMB', 'DESCEND']:
            # Todas las celdas siguen la trend_entrada a excepción de la salida de la última celda
            trend_cell_salida[-1] = trend_salida


    elif flow_trend == 'EVOLUTION':

        if trend_entrada in ['CLIMB', 'DESCEND'] and trend_salida == trend_entrada:
            trend_cell_entrada = [trend_entrada] * num_celdas
            trend_cell_salida = [trend_entrada] * num_celdas

        elif trend_entrada in ['CLIMB', 'DESCEND'] and trend_salida == 'CRUISE':
            # Todas las celdas siguen trend_entrada, excepto la última salida que es CRUISE
            trend_cell_entrada = [trend_entrada] * num_celdas
            trend_cell_salida = [trend_entrada] * num_celdas
            trend_cell_salida[-1] = 'CRUISE'

        elif trend_entrada == 'CRUISE' and trend_salida in ['CLIMB', 'DESCEND']:
            # Todas las celdas siguen trend_salida, excepto la primera entrada que es CRUISE
            trend_cell_entrada = [trend_salida] * num_celdas
            trend_cell_salida = [trend_salida] * num_celdas
            trend_cell_entrada[0] = 'CRUISE'

        elif trend_entrada == 'CLIMB' and trend_salida == 'DESCEND': # Casos especiales - evaluación manual

            if flow == '6_LECMDGL_CL':
                trend_cell_entrada = ['CLIMB','CLIMB','DESCEND']
                trend_cell_salida = ['CLIMB','DESCEND','DESCEND']

            elif flow == '40_LECMDGL_CL':
                trend_cell_entrada = ['CLIMB','DESCEND','DESCEND']
                trend_cell_salida = ['DESCEND','DESCEND','DESCEND']

            elif flow == '44_LECMDGL_CL':
                trend_cell_entrada = ['CLIMB','CLIMB','CLIMB','DESCEND']
                trend_cell_salida = ['CLIMB','CLIMB','DESCEND','DESCEND']

            elif flow == '20_LECMPAL_CL':
                trend_cell_entrada = ['CLIMB','CLIMB','CLIMB','CLIMB','CLIMB','DESCEND']
                trend_cell_salida = ['CLIMB','CLIMB','CLIMB','CLIMB','DESCEND','DESCEND']

            elif flow == '137_LECMPAL_CL':
                trend_cell_entrada = ['CLIMB','CLIMB','CLIMB','CLIMB','DESCEND']
                trend_cell_salida = ['CLIMB','CLIMB','CLIMB','DESCEND','DESCEND']

            elif flow == '115_LECMSAS_CL':
                trend_cell_entrada = ['CLIMB','CLIMB','CLIMB','CLIMB','DESCEND','DESCEND']
                trend_cell_salida = ['CLIMB','CLIMB','CLIMB','DESCEND','DESCEND','DESCEND']

    return trend_cell_entrada, trend_cell_salida


# Aplicar la función fila a fila y asignar a las columnas explicitamente
tendencias = DF_Flujos_completo.apply(calcular_tendencia_celdas, axis=1)
DF_Flujos_completo['Trend_cell_entrada'] = tendencias.apply(lambda x: x[0])
DF_Flujos_completo['Trend_cell_salida'] = tendencias.apply(lambda x: x[1])



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- COMPLETAR EL DATASET DEL ANÁLISIS CELDAS POR FLUJO -------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #

# CÁLCULO DE LA DISTANCIA RECORRIDA, EN MILLAS NÁUTICAS, ASOCIADA A CADA FLUJO CLUSTERIZADO. OBTENCIÓN DE LA DISTANCIA POR CELDA

# DISTANCIA TOTAL RECORRIDA EN EL FLUJO
# Función para calcular la distancia en NM entre dos puntos
def calcular_distancia_flujo_nm(row):
    punto_in = (row['lat_f_in'], row['lon_f_in'])
    punto_out = (row['lat_f_out'], row['lon_f_out'])
    # geopy.distance.geodesic devuelve la distancia en kilómetros
    distancia_km = geodesic(punto_in, punto_out).km
    # Convertir la distancia a millas náuticas (1 NM = 1.852 km) y se redondea a dos decimales
    distancia_NM = round(distancia_km / 1.852, 2)
    return distancia_NM

# Crear la nueva columna 'Distancia_NM'
DF_Flujos_completo['Distancia_flujo_NM'] = DF_Flujos_completo.apply(calcular_distancia_flujo_nm, axis=1)


# DISTANCIA PARCIAL RECORRIDA EN CADA CELDA
# Función para calcular distancias normalizadas por celda
def calcular_distancias_normalizadas_por_celda(row, df_cells):
    flujo_line = row['Line']  # LINESTRING del flujo
    total_distance = row['Distancia_flujo_NM']  # Distancia total del flujo
    distancias_normalizadas = []

    for cell_name in row['Cell_Names']:
        # Obtener el polígono de la celda
        polygon_data = df_cells[df_cells['Cell_Name'] == cell_name]
        if not polygon_data.empty:
            cell_polygon = Polygon(polygon_data.iloc[0]['Coordinates'])

            # Intersección entre flujo y celda
            interseccion = flujo_line.intersection(cell_polygon)

            # Calcular longitud de la intersección como proporción de la distancia total
            if not interseccion.is_empty:
                interseccion_length = interseccion.length  # Longitud en coordenadas de mapa
                proporcion = interseccion_length / flujo_line.length
                distancia_celda = proporcion * total_distance
                distancias_normalizadas.append(round(distancia_celda, 2))  # Redondear a 2 decimales
            else:
                distancias_normalizadas.append(0.00)  # Caso de no intersección
        else:
            distancias_normalizadas.append(0.00)  # Celda no encontrada en DF_cells

    return distancias_normalizadas

# Aplicar la función al dataframe
DF_Flujos_completo['Distancia_por_celda_NM'] = DF_Flujos_completo.apply(lambda row: calcular_distancias_normalizadas_por_celda(row, DF_cells), axis=1)

# # Expandir el anterior dataframe: para cada flujo habrá tantas entradas como celdas de paso lleve asociado el flujo.
# DF_Flujos_completo_exp = DF_Flujos_completo.apply(lambda x: x.explode() if x.name in ['Cell_Names', 'Trend_cell_entrada', 'Trend_cell_salida', 'Distancia_por_celda_NM'] else x, axis=0)

# # Resetear el índice para asegurar que cada fila tenga un índice único
# DF_Flujos_completo_exp = DF_Flujos_completo_exp.reset_index(drop=True)
cols = ['Cell_Names', 'Trend_cell_entrada', 'Trend_cell_salida', 'Distancia_por_celda_NM']


# DF_Flujos_completo_exp = DF_Flujos_completo.copy()
# for col in ['Cell_Names','Trend_cell_entrada','Trend_cell_salida','Distancia_por_celda_NM']:
#     DF_Flujos_completo_exp = DF_Flujos_completo_exp.explode(col)
# DF_Flujos_completo_exp = DF_Flujos_completo_exp.reset_index(drop=True)


# GUARDADO DE AMBOS DATAFRAMES
# Guardado en formato .csv para consulta
DF_Flujos_completo.to_csv(PATH_resultados + 'dataset_celdas_por_flujo_completo.csv', index=False, sep=';')
# DF_Flujos_completo_exp.to_csv(PATH_resultados + 'dataset_celdas_por_flujo_completo_exp.csv', index=False, sep=';')

# Guardado en formato .pkl para su empleo en otros códigos
DF_Flujos_completo.to_pickle(PATH_resultados + 'dataset_celdas_por_flujo_completo.pkl')
# DF_Flujos_completo_exp.to_pickle(PATH_resultados + 'dataset_celdas_por_flujo_completo_exp.pkl')


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------- INSTANTES DE ENTRADA Y SALIDA DE CADA CELDA ------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #

#OBTENCIÓN DEL INSTANTE DE ENTRADA Y DEL INSTANTE DE SALIDA PARA CADA CELDA

#Incluir la información relativa a las actitudes de entrada y salida, y a las distancias totales y por celda
DF_TRAFICO = pd.merge(DF_TRAFICO, DF_Flujos_completo[['Clave_Flujo', 'Sector', 'Flujo_Clusterizado', 'Trend_cell_entrada', 'Trend_cell_salida', 'Distancia_flujo_NM',
                                                               'Distancia_por_celda_NM']], how='left', on=['Clave_Flujo','Sector','Flujo_Clusterizado'])

# Cálculo de la velocidad horizontal del vuelo en cada sector de paso
DF_TRAFICO['VelHoriz_Sector'] = DF_TRAFICO['Distancia_flujo_NM'] / DF_TRAFICO['t-en_sector']

# Cálculo del tiempo de paso por cada una de las celdas. Supongo velocidad constante a lo largo del sector
DF_TRAFICO['lista_t-en_celda'] = DF_TRAFICO.apply(lambda row: [round(dist / row['VelHoriz_Sector'], 2) for dist in row['Distancia_por_celda_NM']], axis=1)

# Cálculo del instante de entrada a cada celda de paso
# Función para calcular las listas con las fechas de entrada y salida para cada celda
def calcular_fechas(row):

    sector_entry_instant = pd.to_datetime(row['sectorEntryInstant'])
    sector_exit_instant = pd.to_datetime(row['sectorExitInstant'])
    tiempos_paso = row['lista_t-en_celda']  # Lista de tiempos de paso por las celdas, en minutos con decimales

    # Inicializar las listas
    fechas_entrada = [sector_entry_instant]
    fechas_salida = []

    # Bucle para calcular el tiempo de entrada y salida por celda
    for t in tiempos_paso:
        tiempo_delta = pd.to_timedelta(t * 60, unit='s') # Convertir tiempo de minutos con decimales a timedelta en segundos
        fecha_salida = fechas_entrada[-1] + tiempo_delta
        fechas_salida.append(fecha_salida)
        fechas_entrada.append(fecha_salida)

    # Ajustar la última fecha de salida al sectorExitInstant
    if len(fechas_salida) > 0:
        fechas_salida[-1] = sector_exit_instant

    # Formatear las fechas en '%Y-%m-%d %H:%M:%S'
    fechas_entrada_formateadas = [fecha.strftime('%Y-%m-%d %H:%M:%S') for fecha in fechas_entrada[:-1]]  # Excluir última entrada duplicada
    fechas_salida_formateadas = [fecha.strftime('%Y-%m-%d %H:%M:%S') for fecha in fechas_salida]

    return fechas_entrada_formateadas, fechas_salida_formateadas


# Aplicar la función para calcular las listas con las fechas de entrada y salida a cada celda
DF_TRAFICO[['lista_fecha-entrada_celda', 'lista_fecha-salida_celda']] = DF_TRAFICO.apply(lambda row: pd.Series(calcular_fechas(row)), axis=1)




#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------- NIVEL DE VUELO DE ENTRADA Y SALIDA DE CADA CELDA --------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# CÁLCULO APROXIMADO DE LOS NIVELES DE VUELO DE ENTRADA Y DE SALIDA A CADA CELDA

# Dataframe de trafico auxiliar para el cálculo de los niveles de vuelo
DF_TRAFICO_ = DF_TRAFICO.copy()
DF_TRAFICO_ = pd.merge(
    DF_TRAFICO_,
    DF_Flujos[['Flujo_Clusterizado', 'lon_f_in', 'lat_f_in', 'lon_f_out', 'lat_f_out']],
    how='left',
    on='Flujo_Clusterizado'
)

# Adaptación de los datos cargados desde csv
DF_cells = DF_cells.copy()
DF_Flujos = DF_Flujos.copy()

DF_cells['Cell_Name'] = DF_cells['Cell_Name'].astype(str).str.strip()

def convertir_polygon(x):
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return None
        return wkt.loads(x)
    return x

DF_cells['Polygon'] = DF_cells['Polygon'].apply(convertir_polygon)

# Funciones para la definición del tipo de trayectoria
def calcular_parabola(punto_in, punto_out, vertice="desconocido", concava_hacia_arriba=True):

    """
    Calcula los coeficientes de una parábola que pasa por dos puntos dados.

    Args:
        punto_in (tuple): Coordenadas (x, y, z) del punto de entrada.
        punto_out (tuple): Coordenadas (x, y, z) del punto de salida.
        vertice (str): Indica si el vértice está en "IN", "OUT", o es "desconocido".
        concava_hacia_arriba (bool): Indica si la parábola es cóncava hacia arriba.

    Returns:
        tuple: Coeficientes de la parábola [a, b, c].
    """

    x_in, y_in, z_in = punto_in
    x_out, y_out, z_out = punto_out

    if vertice == "IN":
        # El vértice es el punto IN. Ajustar el vértice.
        x_vertice = x_in + 1e-6
        y_vertice = y_in + 1e-6
        z_vertice = z_in + 1e-6

    elif vertice == "OUT":
        # El vértice es el punto OUT. Ajustar el vértice.
        x_vertice = x_out - 1e-6
        y_vertice = y_out - 1e-6
        z_vertice = z_out - 1e-6

    else:
        # El vértice es desconocido, asumimos que está en el punto medio
        x_vertice = (x_in + x_out) / 2
        y_vertice = (y_in + y_out) / 2
        z_vertice = max(z_in, z_out) + abs(z_in - z_out) if concava_hacia_arriba else min(z_in, z_out) - abs(z_in - z_out)

    # Resolver el sistema de ecuaciones
    A = np.array([
        [x_in ** 2, x_in, 1],
        [x_vertice ** 2, x_vertice, 1],
        [x_out ** 2, x_out, 1]
    ])
    B = np.array([z_in, z_vertice, z_out])

    coef = np.linalg.solve(A, B)
    return coef  # [a, b, c]


def calcular_recta(punto_in, punto_out):

    """
    Calcula los coeficientes de una recta que pasa por dos puntos dados.

    Args:
        punto_in (tuple): Coordenadas (x, y, z) del punto de entrada.
        punto_out (tuple): Coordenadas (x, y, z) del punto de salida.

    Returns:
        tuple: Coeficientes de la recta [m, n].
    """

    x_in, y_in, z_in = punto_in
    x_out, y_out, z_out = punto_out

    # Evitar división entre cero
    if x_out == x_in:
        return None

    # Calcular la pendiente m
    m = (z_out - z_in) / (x_out - x_in)

    # Calcular el término independiente n
    n = z_in - m * x_in

    coef = (m, n)

    return coef  # [m, n]


# Función para calcular la altitud de un punto según la trayectoria (rectilínea o parabólica)
def calcular_altitud(x, coef, trayectoria):

    if trayectoria == 'rectilinea':
        m, n = coef
        modoC = m * x + n

    elif trayectoria == 'parabolica':
        a, b, c = coef
        modoC = a * x**2 + b * x + c

    return round(modoC, -1)  # Redondear a las decenas


# Función para que las celdas no tengan texto al pasar de pickle a csv
def asegurar_lista(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return [x]
    return []


# Función principal de cálculo de altitud en celda
def calcular_altitudes_en_celdas(df_predicciones, df_flujos, df_cells):
    DF_modoC_cells = pd.DataFrame()

    for _, fila in df_predicciones.iterrows():
        resultados = []
        trayectoria = None

        flujo = fila['Flujo_Clusterizado']
        sector = fila['Sector']
        flightkey = fila['flightKey']
        punto_in = (fila['lon_f_in'], fila['lat_f_in'], fila['modoCIN'])
        punto_out = (fila['lon_f_out'], fila['lat_f_out'], fila['modoCOUT'])
        trend_flow = fila['Flow_Trend']
        trend_entrada = fila['Trend_entrada']
        trend_salida = fila['Trend_salida']

        # CASOS A CONSIDERAR SEGÚN LAS COMBINACIONES DE TREND ENTRADA Y SALIDA EN FUNCIÓN DE LA FLOW TREND: TRAYECTORIA RECTILÍNEA O PARABÓLICA
        # FLOW TREND: CRUISE
        if trend_flow == 'CRUISE' and trend_entrada == 'CRUISE' and trend_salida == 'CRUISE':
            trayectoria = 'rectilinea'

        elif trend_flow == 'CRUISE' and trend_entrada == 'CLIMB' and trend_salida == 'CRUISE':
            trayectoria = 'rectilinea'

        elif trend_flow == 'CRUISE' and trend_entrada == 'DESCEND' and trend_salida == 'CRUISE':
            trayectoria = 'rectilinea'

        # FLOW TREND: EVOLUTION
        elif trend_flow == 'EVOLUTION' and trend_entrada == 'CLIMB' and trend_salida == 'CLIMB':
            trayectoria = 'rectilinea'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'DESCEND' and trend_salida == 'DESCEND':
            trayectoria = 'rectilinea'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'CLIMB' and trend_salida == 'CRUISE':
            trayectoria = 'parabolica'
            concava_hacia_arriba = False
            vertice = 'OUT'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'DESCEND' and trend_salida == 'CRUISE':
            trayectoria = 'parabolica'
            concava_hacia_arriba = True
            vertice = 'OUT'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'CRUISE' and trend_salida == 'CLIMB':
            trayectoria = 'parabolica'
            concava_hacia_arriba = True
            vertice = 'IN'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'CRUISE' and trend_salida == 'DESCEND':
            trayectoria = 'parabolica'
            concava_hacia_arriba = False
            vertice = 'IN'

        elif trend_flow == 'EVOLUTION' and trend_entrada == 'CLIMB' and trend_salida == 'DESCEND':
            trayectoria = 'parabolica'
            concava_hacia_arriba = False
            vertice = 'desconocido'

        # Si no entra en ningún caso, saltar esa fila
        if trayectoria is None:
            continue

        # Calcular coeficientes según la trayectoria (rectilínea o parabólica)
        if trayectoria == 'rectilinea':
            coef = calcular_recta(punto_in, punto_out)
            if coef is None:
                continue

        elif trayectoria == 'parabolica':
            try:
                coef = calcular_parabola(punto_in, punto_out, vertice, concava_hacia_arriba)
            except np.linalg.LinAlgError:
                continue

        # Obtener celdas atravesadas
        celdas = df_flujos.loc[df_flujos['Flujo_Clusterizado'] == flujo, 'Cell_Names']
        if not celdas.empty:
            celdas = asegurar_lista(celdas.iloc[0])
        else:
            celdas = []

        for celda in celdas:
            celda = str(celda).strip()

            polygon_row = df_cells.loc[df_cells['Cell_Name'] == celda, 'Polygon']
            if polygon_row.empty:
                print(f"Celda no encontrada en df_cells: {celda} | flujo={flujo} | flightKey={flightkey}")
                continue

            polygon = polygon_row.iloc[0]

            if polygon is None:
                continue

            line = LineString([
                (fila['lon_f_in'], fila['lat_f_in']),
                (fila['lon_f_out'], fila['lat_f_out'])
            ])

            interseccion = line.intersection(polygon)

            if interseccion.is_empty:
                continue

            if interseccion.geom_type == 'LineString':
                puntos = [interseccion.coords[0], interseccion.coords[-1]]

            elif interseccion.geom_type == 'MultiLineString':
                puntos = []
                for tramo in interseccion.geoms:
                    puntos.extend([tramo.coords[0], tramo.coords[-1]])

            elif interseccion.geom_type == 'Point':
                puntos = [interseccion.coords[0]]

            elif interseccion.geom_type == 'MultiPoint':
                puntos = [pt.coords[0] for pt in interseccion.geoms]

            else:
                continue

            for x, y in puntos:
                altitud = calcular_altitud(x, coef, trayectoria)
                resultados.append({
                    'flightKey': flightkey,
                    'Sector': sector,
                    'Flujo_Clusterizado': flujo,
                    'Cell_Name': celda,
                    'lon': x,
                    'lat': y,
                    'Altitud': altitud
                })

        if len(resultados) == 0:
            continue

        resultados_DF = pd.DataFrame(resultados)

        # Reorganizar los datos del DF. Crear columnas separadas para coordenadas "in" y "out" de los puntos que definen el flujo clusterizado en cada celda
        resultados_in = resultados_DF.groupby('Cell_Name').nth(0).reset_index(drop=True)
        resultados_out = resultados_DF.groupby('Cell_Name').nth(1).reset_index(drop=True)

        resultados_in.columns = [
            'flightKey', 'Sector', 'Flujo_Clusterizado', 'Cell_Name',
            'lon_cell_in', 'lat_cell_in', 'modoCIN_cell'
        ]

        resultados_out.columns = [
            'flightKey', 'Sector', 'Flujo_Clusterizado', 'Cell_Name',
            'lon_cell_out', 'lat_cell_out', 'modoCOUT_cell'
        ]

        # Elimino columna 'flujo' repetida en ambos dataframes
        resultados_out = resultados_out.drop(columns=['flightKey', 'Sector', 'Flujo_Clusterizado'])

        # Unir ambos dataFrames por la columna 'celda'
        resultados_modoC = pd.merge(resultados_in, resultados_out, on='Cell_Name', how='left')

        # Reorganizar columnas
        resultados_modoC = resultados_modoC[
            ['flightKey', 'Sector', 'Flujo_Clusterizado', 'Cell_Name',
             'lon_cell_in', 'lat_cell_in', 'lon_cell_out', 'lat_cell_out',
             'modoCIN_cell', 'modoCOUT_cell']
        ]

        # Concatenar los dataframes generados para cada fila
        DF_modoC_cells = pd.concat([DF_modoC_cells, resultados_modoC], ignore_index=True)

    return DF_modoC_cells


# Aplicar la función principal al dataframe de predicciones
DF_resultados_modoC = calcular_altitudes_en_celdas(DF_TRAFICO_, DF_Flujos, DF_cells)


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------ DATASET DE TRÁFICO COMPLETADO Y GUARDADO -------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# DF DE TRÁFICO A NIVEL DE CELDA: EXPANSIÓN DEL DF_TRAFICO Y UNIÓN DE LOS RESULTADOS OBTENIDOS PARA LOS NIVELES DE VUELO

# ELIMINACIÓN DE COLUMNAS INNECESARIAS DEL DF DE PREDICCIONES
DF_TRAFICO.drop('Secuencia_Sectores', axis=1, inplace=True)
DF_TRAFICO.drop('Secuencia_Flujos', axis=1, inplace=True)
DF_TRAFICO.drop('Secuencia_Claves_Flujos', axis=1, inplace=True)
DF_TRAFICO.drop('Secuencia_Flow_Trend', axis=1, inplace=True)
DF_TRAFICO.drop('VelHoriz_Sector', axis=1, inplace=True)
DF_TRAFICO.drop('Distancia_flujo_NM', axis=1, inplace=True)
DF_TRAFICO.drop('Distancia_por_celda_NM', axis=1, inplace=True)


# EXPANSIÓN DEL DF
# Lista de columnas consideradas en la expansión
list_columns = ['Cell_Names', 'Trend_cell_entrada', 'Trend_cell_salida', 'lista_t-en_celda', 'lista_fecha-entrada_celda', 'lista_fecha-salida_celda']

# Repetir filas según la longitud de las listas en la columna 'Cell_Names'
DF_TRAFICO_exploded = DF_TRAFICO.loc[DF_TRAFICO.index.repeat(DF_TRAFICO['Cell_Names'].str.len())].reset_index(drop=True)

# Expandir las columnas de listas para distribuir los valores
for col in list_columns:
    DF_TRAFICO_exploded[col] = DF_TRAFICO[col].explode(ignore_index=True)

# Resetear índices del nuevo DF de predicciones expandido
DF_TRAFICO_exploded.reset_index(drop=True, inplace=True)

# Renombrar columnas del dataframe expandido
DF_TRAFICO_exploded.rename(columns={'Cell_Names': 'Cell_Name'}, inplace=True)
DF_TRAFICO_exploded.rename(columns={'lista_t-en_celda': 't-en_celda'}, inplace=True)
DF_TRAFICO_exploded.rename(columns={'lista_fecha-entrada_celda': 'fecha-entrada_celda'}, inplace=True)
DF_TRAFICO_exploded.rename(columns={'lista_fecha-salida_celda': 'fecha-salida_celda'}, inplace=True)


# INCORPORACIÓN DE LOS RESULTADOS OBTENIDOS DEL CÁLCULO APROXIMADO DE LOS FL DE ENTRADA Y SALIDA POR CELDA
DF_TRAFICO_CELDA = pd.merge(DF_TRAFICO_exploded, DF_resultados_modoC, how='left', on=['flightKey', 'Sector', 'Flujo_Clusterizado', 'Cell_Name'])

# Reordenar las columnas del dataframe final
DF_TRAFICO_CELDA = DF_TRAFICO_CELDA[['flightKey', 'origen_destino', 'airline', 'aircraftType', 'wake', 'routeType', 'ETOT', 'IOBT', 'Secuencia', 'Sector',
                                     'Flujo_Clusterizado','Clave_Flujo','Cell_Name','Flow_Trend','Trend_cell_entrada','Trend_cell_salida','fecha-entrada_celda',
                                     'fecha-salida_celda','t-en_celda','modoCIN_cell','modoCOUT_cell','lon_cell_in','lat_cell_in','lon_cell_out','lat_cell_out',
                                     'Trend_entrada','Trend_salida','sectorEntryInstant','sectorExitInstant','modoCIN','modoCOUT','attitudIN','attitudOUT',
                                     't-hasta_sector','t-en_sector','t-salida_sector']]


# GUARDAR EL DATAFRAME FINAL
# if tipo_trafico == 'predicciones':
#     DF_TRAFICO_CELDA.to_csv(PATH_TRAFICO_CELDA + f'\\DF_PREDICCIONES_CELDA.csv', index=False, sep=';')
#     DF_TRAFICO_CELDA.to_pickle(PATH_TRAFICO_CELDA + f'\\DF_PREDICCIONES_CELDA.pkl')
# elif tipo_trafico == 'real':
DF_TRAFICO_CELDA.to_csv(PATH_TRAFICO_CELDA + f'\\DF_T_REAL_CELDA.csv', index=False, sep=';')
DF_TRAFICO_CELDA.to_pickle(PATH_TRAFICO_CELDA + f'\\DF_T_REAL_CELDA.pkl')
    
    
    
    
    
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
