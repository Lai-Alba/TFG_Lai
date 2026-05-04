#%% SABs TIPO 1
# Para mostrar el mapa de flujos
# identificar flujos paralelos a la frontera -> sabs tipo 1 
#


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
from matplotlib.patches import Patch
import geopy.distance
from geopy.distance import geodesic
from shapely.wkt import loads
import pickle
from shapely.geometry import Polygon, Point, LineString, box, MultiLineString 
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
import time
from shapely.ops import unary_union
start_time = time.time()


### DIRECTORIOS - ACC Madrid Norte
PATH_TRAFICO = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\Junio2022\\'
PATH_SECTOR_DATA = 'F:\\Users\\Lai\\Datos\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'
PATH_flujos = 'F:\\Users\\Lai\\original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\'
PATH_resultados = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\Junio2022\\test\\'
PATH_TRAFICO_CELDA = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\mallado_20x20\\'
PATH_bordes = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\'
PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\'

### Parámetros de estudio

# configuración de estudio
configuracion_estudio = 'CNF5A'

# Día seleccionado
fecha_sel = input("Introduce un día del junio de 2022 (2022-06-DD): ")

# Tamaño de celda en nm
cell_size_nm = 20
# cell_size_nm = input("Introduce el tamaño de celda en NM (ej. 20): ")

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

## -------------------------------------------------------------------------------------------------------------------- #

# DATASET ANÁLISIS FLUJOS POR CELDA: qué flujos atraviesan cada celda
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo.pkl')

# DATASET DE TRÁFICO REAL: qué vuelos reales hay en el día seleccionado
DF_Trafico = pd.read_pickle(PATH_TRAFICO + f'dataset_vuelos_reales_2022-06-{fecha_sel}.pkl')

# DATASET DE CELDAS FRONTERIZAS: qué celdas están en la frontera entre sectores
df_border_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_border_cells_{cell_size_nm}.pkl')

# DATASET DE CELDAS DEL MALLADO: qué celdas hay en el mallado (con su geometría)
mallado_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_DF_cells.pkl')



# --- INICIO DEL BLOQUE DE FILTRADO ---
# 1. Cargar el archivo de referencia (vuelos reales del día 01)
df_referencia = pd.read_pickle(PATH_TRAFICO_CELDA + f'DF_T_REAL_CELDA_20x20_{fecha_sel}.pkl')

# 2. Cargar los datasets que queremos filtrar
# NOTA: iguales que anteriores pero son apra que no influyen a los anteriores datasets
df_celdas_por_flujo_raw = pd.read_pickle(PATH_resultados + 'dataset_celdas_por_flujo.pkl')
df_flujos_por_celda_raw = pd.read_pickle(PATH_resultados + 'dataset_flujos_por_celda.pkl')

# 3. Extraer los flujos únicos del día 01
flujos_dia_clave = set(df_referencia['Clave_Flujo'].unique())
flujos_dia_cluster = set(df_referencia['Flujo_Clusterizado'].unique())

# 4. Filtrar 'dataset_celdas_por_flujo'
# Nos quedamos solo con las filas cuyos flujos existen en el archivo del día 01
DF_Flujos = df_celdas_por_flujo_raw[
    df_celdas_por_flujo_raw['Clave_Flujo'].isin(flujos_dia_clave)
].copy()

# 5. Filtrar 'dataset_flujos_por_celda'
# def filtrar_lista_texto(texto_lista):
#     try:
#         # Convertir el string del CSV en lista real
#         lista = ast.literal_eval(texto_lista)
#         # Solo mantener flujos que están en el día 01
#         return [f for f in lista if f in flujos_dia_cluster]
#     except:
#         return []

# df_flujos_por_celda_raw['Flujos_Clusterizados'] = df_flujos_por_celda_raw['Flujos_Clusterizados'].apply(filtrar_lista_texto)

# Eliminar celdas que no tengan flujos tras el filtro y guardar en DF_cells
DF_cells = df_flujos_por_celda_raw[
    df_flujos_por_celda_raw['Flujos_Clusterizados'].map(len) > 0
].copy()

print(f"Las celdas con los flujos filtrados son: {DF_cells}")

# Opcional: convertir de vuelta a string si necesitas el formato original para guardar
# DF_cells['Flujos_Clusterizados'] = DF_cells['Flujos_Clusterizados'].astype(str)

print(f"Filtrado listo: {len(DF_Flujos)} flujos y {len(DF_cells)} celdas activas para el día 01.")
# --- FIN DEL BLOQUE DE FILTRADO ---
print(DF_Flujos)
# A partir de aquí, el resto de tu código usará DF_Flujos y DF_cells ya filtrados.

DF_Flujos.to_csv(PATH_resultados + f'dataset_celdas_por_flujo_{fecha_sel}_filtrado.csv', index=False)
DF_cells.to_csv(PATH_resultados + f'dataset_flujos_por_celda_{fecha_sel}_filtrado.csv', index=False)

DF_Flujos.to_pickle(PATH_resultados + f'dataset_celdas_por_flujo_{fecha_sel}_filtrado.pkl')
DF_cells.to_pickle(PATH_resultados + f'dataset_flujos_por_celda_{fecha_sel}_filtrado.pkl')

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

DF_info_conf.to_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')
DF_info_conf.to_csv(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.csv', index=False)

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


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- REPRESENTACIÓN DE LAS TRAYECTORIAS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

## -------------------------------------------------------------------------------------------------------------------- #
gdf_cells = gpd.GeoDataFrame(mallado_cells.copy(), geometry='Polygon')
gdf_cells = gdf_cells.set_geometry('Polygon')

# # Crear una figura y un eje para la gráfica
# fig, ax_3 = plt.subplots(figsize=(20, 10)) # añadido tamaño para que se vea mejor

# # Graficar las celdas del mallado (polígonos)
# for _, celda in gdf_cells.iterrows():
#     x, y = celda['Polygon'].exterior.xy
#     ax_3.fill(x, y, alpha=0.3, color='lightblue', edgecolor='blue')
#     # Colocar nombre de la celda en el centro del polígono
#     # ax_3.text(celda['Polygon'].centroid.x, celda['Polygon'].centroid.y, celda['Cell_Name'],fontsize=4, ha='center', color='black')

# # 2. Graficar TODOS los flujos (iterando sobre el DataFrame)
# for idx, flujo in DF_Flujos.iterrows():
#     x_flujo, y_flujo = flujo['Line'].xy
#     # Usamos un alpha bajo (0.5) para que los flujos superpuestos se vean más oscuros
#     ax_3.plot(x_flujo, y_flujo, color='red', linewidth=1, alpha=0.5)
    
#     # Opcional: Colocar el nombre de cada flujo (CUIDADO si hay muchos, puede saturar el gráfico)
#     # Si quieres ponerlos, descomenta la línea de abajo:
#     # ax_3.text(flujo['Line'].centroid.x, flujo['Line'].centroid.y, flujo['Flujo_Clusterizado'], fontsize=6, ha='center', color='darkred')

# # for cell_name in df_border_cells['Cell_Name']:
# #     poly = gdf_cells[gdf_cells['Cell_Name'] == cell_name]['Polygon'].iloc[0]
# #     x, y = poly.exterior.xy
# #     ax_3.plot(x, y, color='black', linewidth=2.0)

# #PLOTEAR EL ACC

# for index, row in DF_info_conf.iterrows():
#     poligono = row['Contorno Sector']
#     x, y = poligono.exterior.xy
#     ax_3.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'{row["SECTOR_ID"]}')

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
# plt.show()


# # Personalizar el gráfico
# ax_3.set_xlim(min_lon, max_lon)
# ax_3.set_ylim(min_lat, max_lat)
# ax_3.set_title('REPRESENTACIÓN DE TODOS LOS FLUJOS SOBRE EL MALLADO del día ' + fecha_sel)
# ax_3.set_xlabel('LONGITUD [º]')
# ax_3.set_ylabel('LATITUD [º]')
# ax_3.set_aspect('equal')


# plt.show()


# 1. ASEGURAR QUE TODOS SON GEODATAFRAMES
# Esto nos permite graficar todo de golpe sin usar bucles for
gdf_cells = gpd.GeoDataFrame(mallado_cells.copy(), geometry='Polygon')
gdf_flujos = gpd.GeoDataFrame(DF_Flujos.copy(), geometry='Line')
gdf_sectores = gpd.GeoDataFrame(DF_info_conf.copy(), geometry='Contorno Sector')

# 2. CREAR FIGURA Y EJES
fig, ax_3 = plt.subplots(figsize=(15, 10)) # Un poco más de altura para la leyenda inferior

# 3. GRAFICAR MALLADO (Capa base)
gdf_cells.plot(ax=ax_3, facecolor='lightblue', edgecolor='blue', alpha=0.2, zorder=1)

# 4. GRAFICAR SECTORES (Capa intermedia)
# Al usar column='SECTOR_ID' y un colormap ('tab20'), GeoPandas colorea cada sector automáticamente
gdf_sectores.plot(ax=ax_3, column='SECTOR_ID', cmap='tab20', 
                  edgecolor='black', linewidth=1.5, alpha=0.5, 
                  legend=True, zorder=2)

# Añadir el nombre del sector en su centroide para ubicarlo fácilmente
for _, row in gdf_sectores.iterrows():
    centro = row['Contorno Sector'].centroid
    ax_3.text(centro.x, centro.y, row['SECTOR_ID'], 
              fontsize=10, fontweight='bold', ha='center', va='center', 
              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1), zorder=4)

# 5. GRAFICAR FLUJOS (Capa superior)
gdf_flujos.plot(ax=ax_3, color='red', linewidth=1, alpha=0.4, zorder=3)

# 6. CONSTRUIR LEYENDA UNIFICADA
# GeoPandas ya generó una leyenda para los sectores gracias a `legend=True`. 
# Vamos a capturarla y añadirle manualmente el mallado y los flujos.
handles, labels = ax_3.get_legend_handles_labels()

# Añadimos los elementos faltantes a las listas de la leyenda
# handles.append(Patch(facecolor='lightblue', edgecolor='blue', alpha=0.3))
# labels.append('Mallado (Celdas)')

handles.append(Line2D([0], [0], color='red', lw=1.5, alpha=0.5))
labels.append('Flujos (Trayectorias)')

# Volvemos a generar la leyenda con todos los elementos combinados
ax_3.legend(handles, labels, loc='upper right', 
            ncol=2, fontsize='medium', title='Elementos del Mapa', title_fontsize='large')
# 7. PERSONALIZACIÓN DE EJES Y TÍTULO
# Nota: Si min_lon y max_lon no están definidos, puedes usar el bounds de los sectores:
# minx, miny, maxx, maxy = gdf_sectores.total_bounds
# ax_3.set_xlim(minx, maxx)
# ax_3.set_ylim(miny, maxy)

ax_3.set_xlim(min_lon, max_lon)
ax_3.set_ylim(min_lat, max_lat)
ax_3.set_title(f'Representación de todos los flujos sobre el mallado\nFecha: {fecha_sel}-06-2022', 
               fontsize=16, fontweight='bold', pad=15)
ax_3.set_xlabel('LONGITUD [º]', fontsize=12)
ax_3.set_ylabel('LATITUD [º]', fontsize=12)
ax_3.set_aspect('equal')
ax_3.grid(True, linestyle=':', alpha=0.6)

# Aseguramos que los márgenes se ajusten bien antes de mostrar
plt.tight_layout()
plt.show()

#%%

# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- DETECCIÓN DE SAB TIPO 1 (PARALELOS) ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# --- 1. Obtener las fronteras entre sectores ---
# Creamos una lista de todos los polígonos de sectores en la configuración
sector_polygons = DF_info_conf['Contorno Sector'].tolist()
print(sector_polygons)

# Para encontrar fronteras internas, buscamos la intersección de los límites de cada par de sectores
fronteras_internas = []
for i, poly1 in enumerate(sector_polygons):
    for j, poly2 in enumerate(sector_polygons):
        if i < j:
            inter = poly1.boundary.intersection(poly2.boundary)
            if not inter.is_empty:
                if isinstance(inter, (LineString, MultiLineString)):
                    fronteras_internas.append(inter)

# Unimos todas las líneas de frontera en un solo objeto
fronteras_totales = unary_union(fronteras_internas)

# --- 2. Parámetros de filtrado ---
DISTANCIA_PROXIMIDAD = 0.2    # Grados (aprox 5-6 NM) para considerar "cerca"
TOLERANCIA_ANGULO = 5      # Grados de diferencia máxima para considerar "paralelo"

def calcular_angulo_linea(line):
    """Calcula el ángulo aproximado manejando tanto LineString como MultiLineString."""
    if line.is_empty:
        return 0
    
    # Si es MultiLineString, extraemos las coordenadas de sus partes
    if hasattr(line, 'geoms'):
        coords = []
        for part in line.geoms:
            coords.extend(list(part.coords))
    else:
        coords = list(line.coords)
        
    if len(coords) < 2: 
        return 0
        
    p1, p2 = coords[0], coords[-1]
    # Usamos LaTeX para la fórmula matemática:
    # $\theta = \arctan2(\Delta y, \Delta x)$
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180

# --- 3. Buscar flujos cercanos y paralelos ---
buffer_fronteras = fronteras_totales.buffer(DISTANCIA_PROXIMIDAD)
sab_tipo1_flows = []
fronteras_cercanas = []

for idx, flujo in DF_Flujos.iterrows():
    linea_flujo = flujo['Line']
    nombre_flujo = flujo['Flujo_Clusterizado']

    # ¿Está cerca de alguna frontera?
    if linea_flujo.intersects(buffer_fronteras):
        # Calculamos ángulos para verificar paralelismo
        angulo_flujo = calcular_angulo_linea(linea_flujo)
        
        # Comparamos con las fronteras cercanas
        es_paralelo = False
        frontera_detectada = None

        for frontera in fronteras_internas:
            if linea_flujo.distance(frontera) < DISTANCIA_PROXIMIDAD:
                # La función ahora ya maneja MultiLineString internamente
                angulo_front = calcular_angulo_linea(frontera)
                diff = abs(angulo_flujo - angulo_front)
                
                # Comprobamos paralelismo con tolerancia
                if diff < TOLERANCIA_ANGULO or diff > (180 - TOLERANCIA_ANGULO):
                    es_paralelo = True
                    frontera_detectada = frontera
                    break
        
        if es_paralelo:
            sab_tipo1_flows.append(flujo['Flujo_Clusterizado'])
            fronteras_cercanas.append({
                'Flujo_Clusterizado': nombre_flujo,
                'Frontera_Interna': frontera_detectada
            })

print(f"Flujos SAB Tipo 1 detectados (paralelos a fronteras): {sab_tipo1_flows}")
print(f"Fronteras cercanas a SAB Tipo 1: {fronteras_cercanas[0]['Frontera_Interna']}")
df_fronteras_relacionadas = pd.DataFrame(fronteras_cercanas)

# --- 1. CONFIGURACIÓN DE PARÁMETROS ---
DISTANCIA_EXTREMO = 0.01  # Distancia para considerar que están "unidos" (~1.2 NM)
MAX_INTERSECCIONES = 0  # Límite máximo de cruces permitidos (0, 1 o 2)
MAX_CONEXIONES_EXTREMOS = 3  # Límite máximo de uniones en extremos permitidas

# Extraer todas las trayectorias del día para comparar
geometrias_dia = DF_Flujos['Line'].tolist()
nombres_dia = DF_Flujos['Flujo_Clusterizado'].tolist()
geometria_sab = [DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == f, 'Line'].iloc[0] for f in sab_tipo1_flows]

# B. Extraer SOLO las trayectorias SAB identificadas (para el Criterio 1 - Cruces entre sí)
geometrias_sab = []
nombres_sab = []
for nombre_sab in sab_tipo1_flows:
    linea = DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == nombre_sab, 'Line'].iloc[0]
    geometrias_sab.append(linea)
    nombres_sab.append(nombre_sab)

# --- 2. FILTRADO DE FLUJOS SAB TIPO 1 ---
sab_finales_criterios = []

for nombre_f in sab_tipo1_flows:
    # Obtener la geometría del flujo SAB evaluado
    linea_evaluada = DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == nombre_f, 'Line'].iloc[0]
    p_inicio = Point(linea_evaluada.coords[0])
    p_final = Point(linea_evaluada.coords[-1])
    
    cruces_entre_sabs = 0
    flujos_sab_intersecados = [] 
    flujos_unidos_dia = []       

    # --- CRITERIO 1: Intersecciones SOLAMENTE ENTRE FLUJOS SABs ---
    for nombre_otro_sab, linea_otro_sab in zip(nombres_sab, geometrias_sab):
        if nombre_f == nombre_otro_sab: # No ev0aluarlo contra sí mismo
            continue
            
        interseccion = linea_evaluada.intersection(linea_otro_sab)
        if not interseccion.is_empty:
            puntos_cruce = 0
            if isinstance(interseccion, Point):
                puntos_cruce = 1
            elif hasattr(interseccion, 'geoms'):
                puntos_cruce = len([g for g in interseccion.geoms if isinstance(g, Point)])
            
            if puntos_cruce > 0:
                cruces_entre_sabs += puntos_cruce
                if nombre_otro_sab not in flujos_sab_intersecados:
                    flujos_sab_intersecados.append(nombre_otro_sab)

    # --- CRITERIO 2: Uniones en los extremos con TODOS LOS FLUJOS DEL DÍA ---
    for nombre_otro_dia, linea_otro_dia in zip(nombres_dia, geometrias_dia):
        if nombre_f == nombre_otro_dia:
            
            continue
            
        if linea_otro_dia.distance(p_inicio) < DISTANCIA_EXTREMO or \
           linea_otro_dia.distance(p_final) < DISTANCIA_EXTREMO:
            if nombre_otro_dia not in flujos_unidos_dia:
                flujos_unidos_dia.append(nombre_otro_dia)
                
    # --- APLICAR CONDICIONES FINALES ---
    num_uniones = len(flujos_unidos_dia)
    
    # Queremos: Cruces (solo entre SABs) <= 2 Y Uniones (con flujos del día) <= 5
    if (cruces_entre_sabs <= MAX_INTERSECCIONES) and (num_uniones <= MAX_CONEXIONES_EXTREMOS):
        sab_finales_criterios.append(nombre_f)
        print(f"-> SAB '{nombre_f}' aceptado: {cruces_entre_sabs} cruces SAB, {num_uniones} conexiones de extremo.")
    else:
        # Debugging opcional: ver por qué se rechazan
        print(f"-> SAB '{nombre_f}' descartado: {cruces_entre_sabs} cruces SAB, {num_uniones} conexiones de extremo.")
        pass

# --- 3. ACTUALIZACIÓN DE LAS CELDAS SAB TIPO 1 ---

df_fronteras_filtrado = df_fronteras_relacionadas[
    df_fronteras_relacionadas['Flujo_Clusterizado'].isin(sab_finales_criterios)
].copy()

celdas_sab_tipo1 = {'Cell_Name': [], 'Flujo': [], 'Frontera_Interna': []}
flujos_def = []

for _, celda in DF_cells.iterrows():
    flujos_en_celda = celda['Flujos_Clusterizados']
    lista_f = ast.literal_eval(flujos_en_celda) if isinstance(flujos_en_celda, str) else flujos_en_celda
    
    # Buscamos los flujos que coinciden
    flujo_encontrado = set(lista_f).intersection(set(sab_finales_criterios))
    
    if flujo_encontrado:
        print(f"Flujos encontrados en la celda '{celda['Cell_Name']}': {flujo_encontrado}")
        
        # if celda['Polygon'].intersects(buffer_fronteras):
            # Ahora sí puedes usar las claves 'Cell_Name' y 'Flujo'
        celdas_sab_tipo1['Cell_Name'].append(celda['Cell_Name'])
            
            # Convertimos el set de flujos a un string para guardarlo
        flujos_texto = ", ".join(list(flujo_encontrado))
        celdas_sab_tipo1['Flujo'].append(flujos_texto)

        fronteras_asociadas = df_fronteras_filtrado[
            df_fronteras_filtrado['Flujo_Clusterizado'].isin(flujo_encontrado)
        ]['Frontera_Interna'].tolist()
        
        # Si hay varias, las guardamos como una lista o podrías unificarlas
        celdas_sab_tipo1['Frontera_Interna'].append(fronteras_asociadas)           

# --- REPORTE FINAL ---
print(f"Filtrado estricto completado:")
print(f"- Flujos SAB originales: {len(sab_tipo1_flows)}")
print(f"- Flujos SAB válidos (<=2 cruces Y extremos aislados): {len(sab_finales_criterios)}")
print(f"- Celdas SAB Tipo 1 resultantes: {celdas_sab_tipo1['Cell_Name']}")

print(celdas_sab_tipo1['Frontera_Interna'])
print(sab_finales_criterios)

df_celdas_sab_1 = pd.DataFrame(celdas_sab_tipo1)    
df_celdas_sab_1.to_csv(PATH_sabs + f'celdas_sab_tipo1_2022-06-{fecha_sel}.csv', index=False, encoding='latin1')
df_celdas_sab_1.to_pickle(PATH_sabs + f'celdas_sab_tipo1_2022-06-{fecha_sel}.pkl')

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------- VISUALIZACIÓN DE FLUJOS DETECTADOS Y CELDAS SAB TIPO 1 --------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import matplotlib.patches as mpatches

# 1. Crear la figura
fig, ax_res = plt.subplots(figsize=(15, 10))

# 2. Dibujar los Sectores del ACC (Polígonos de fondo)
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    ax_res.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, label=f"Sector: {row['SECTOR_ID']}")
    # Opcional: Añadir nombre del sector en el centro
    ax_res.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], fontsize=10, ha='center', weight='bold')

# 3. Dibujar TODAS las celdas del mallado (opcional, en gris muy tenue)
for _, celda in gdf_cells.iterrows():
    x, y = celda['Polygon'].exterior.xy
    ax_res.plot(x, y, color='gray', alpha=0.1, linewidth=0.5)

# 4. Resaltar las CELDAS identificadas como SAB Tipo 1 (en Amarillo)
# Usamos el set de celdas_sab_tipo1 que calculamos en el paso anterior
for cell_name in celdas_sab_tipo1['Cell_Name']:
    # Buscamos la geometría de la celda en el GeoDataFrame original
    poly_match = gdf_cells[gdf_cells['Cell_Name'] == cell_name]
    if not poly_match.empty:
        poly = poly_match['Polygon'].iloc[0]
        x, y = poly.exterior.xy
        mitad = len(x) // 2
        ax_res.fill(x, y, color='yellow', alpha=0.5, edgecolor='orange', linewidth=1, zorder=3)
        # ax_res.text(x[mitad], y[mitad], cell_name, 
        #             color='darkred', fontsize=9, fontweight='bold',
        #             ha='center', va='center', zorder=5,
        #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

# 5. Dibujar los FLUJOS PARALELOS detectados (en Rojo Intenso)
for idx, flujo in DF_Flujos.iterrows():
    nombre_flujo = flujo['Flujo_Clusterizado']
    x_f, y_f = flujo['Line'].xy
    
    if nombre_flujo in sab_finales_criterios:
        # Dibujar la línea del flujo SAB
        ax_res.plot(x_f, y_f, color='red', linewidth=2.5, alpha=0.9, zorder=4)
        
        # Calcular el punto medio de la línea para colocar el texto
        mitad = len(x_f) // 2
        
        # Añadir el nombre del flujo
        ax_res.text(x_f[mitad], y_f[mitad], nombre_flujo, 
                    color='darkred', fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
    else:
        # Dibujar el resto de flujos en azul/gris para dar contexto
        ax_res.plot(x_f, y_f, color='blue', linewidth=0.5, alpha=0.2, zorder=2)


# 6. Configuración estética del gráfico
ax_res.set_xlim(min_lon, max_lon)
ax_res.set_ylim(min_lat, max_lat)
ax_res.set_aspect('equal')
ax_res.set_title(f'Flujos paralelos identificados\nFecha: {fecha_sel}-06-2022', fontsize=14)
ax_res.set_xlabel('Longitud [º]')
ax_res.set_ylabel('Latitud [º]')

# Crear una leyenda personalizada
patch_sab = mpatches.Patch(color='yellow', alpha=0.5, label='Celdas que atraviesan los flujos')
line_flow = Line2D([0], [0], color='red', linewidth=2.5, label='Flujos Paralelos Detectados')
line_context = Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.4, label='Otros flujos del día')

ax_res.legend(handles=[patch_sab, line_flow, line_context], loc='upper right', frameon=True)

plt.tight_layout()
plt.show()


# %%
