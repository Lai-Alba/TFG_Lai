# SAB_tipo1
# 
#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

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
PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\'
PATH_COMPLEJIDAD = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Junio2022\\test\\' # Asegúrate de que esta ruta es correcta
PATH_resultados = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\Junio2022\\test\\'

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
sab_tipo1 = pd.read_pickle(PATH_sabs + f'celdas_sab_tipo1_2022-06-{fecha_sel}.pkl')
sab_tipo2 = pd.read_pickle(PATH_sabs + f'celdas_sab_tipo2_suma_2022-06-{fecha_sel}.pkl')

# Datos de los sabs tipo 2 agrupados y refinados
grupo_sab2 = pd.read_pickle(PATH_sabs + f'SABs_Suma_Diaria_agrupados_2022-06-{fecha_sel}.pkl')

# Datos del mallado de tamaño 20x20 nm
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')

# Datos de los sectores (con su geometría)
DF_info_conf = pd.read_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')

## Procesado para obtener Geodataframes------------------------------------------------------------------------------- 

gdf_sab2 = gpd.GeoDataFrame(sab_tipo2, geometry='geometry')
gdf_sab2_aprup = gpd.GeoDataFrame(grupo_sab2, geometry='geometry')
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
# ------------------------------------------------------------------------------------ #
# -------------------------------- EVALUACIÓN DE SABS -------------------------------- #
# ------------------------------------------------------------------------------------ #

# Las celdas de SABs de cada tipo están calculadas en los archivos SAB_tipo1 y SAB_tipo2. 
# Sin embargo estas sabs deben filtrarse porque pueden coincidirse entre sí. Por lo cual,
# en esta sección se filtra las celdas coincidentes y evaluar esas celdas para ver si
# hay que conservar esas celdas coincidentes e identificar qué tipo de sab son porque 
# el tipo influye el método de diseño de la forma de sabs.

# 1. se evalua las celdas del tipo 1 que tiene un nivel alto de complejidad
# 2. se elimina las celdas del tipo 2 que coinciden con el tipo 1,
# pero conserva las celdas del tipo 2 que conectan con las del tipo 1, 
# y que tenga un tamaño importante (ej: más de 3 celdas conectadas) 

## Funciones para la evaluación de sabs----------------------------------------------------

# Función para evaluar las celdas del tipo 1 por su complejidad
def filtrar_sabs_por_complejidad(lista_evaluar, df_complejidad, umbral_maximo):
    """
    Filtra las celdas de Tipo 1. Si superan la complejidad máxima, se descartan.
    
    Args:
        lista_evaluar (list): Lista de IDs de celdas (ej. ['Cell_77', 'Cell_200'])
        df_complejidad (DataFrame): El DataFrame que contiene la columna 'Complejidad_Total_Dia'
        umbral_maximo (float): El valor límite de complejidad permitido.
        
    Returns:
        list: Solo las celdas que pasaron la evaluación (complejidad <= umbral).
    """
    # 1. Filtramos el DataFrame original para quedarnos solo con las celdas a evaluar
    # Nota: Usamos 'Cell_Name' o la columna correspondiente donde estén los IDs
    df_pendientes = df_complejidad[df_complejidad['Cell_Name'].isin(lista_evaluar)]
    print(f"La complejidad de las celdas a evaluar es:\n{df_pendientes[['Cell_Name', 'Complejidad_Total_Dia']].to_string(index=False)}")
    
    # 2. Aplicamos la lógica: se mantienen si complejidad <= umbral_maximo
    sabs_validados = df_pendientes[df_pendientes['Complejidad_Total_Dia'] <= umbral_maximo]
    
    # 3. Retornamos la lista de los que sí cumplen
    return list(sabs_validados['Cell_Name'])

# Función para que una vez identificadas las celdas inválidas, se identique los grupos 
# que contienen esas celdas descartadas

def grupo_es_valido(lista_celdas_grupo, celdas_permitidas):
    # Convertimos la lista de la fila a set para una operación rápida
    # Si la intersección NO está vacía, el grupo contiene celdas válidas
    return not set(lista_celdas_grupo).isdisjoint(celdas_permitidas)

## Proceso de evaluación-------------------------------------------------------------

# Definimos un umbral, por ejemplo: 10.0
UMBRAL = 30.0

# 1. Identificar celdas iniciales y coincidencias
tipo1_cells = set(sab_tipo1['Cell_Name'])
tipo2_cells = set(sab_tipo2['Cell_Name'])
coincidencias = tipo1_cells.intersection(tipo2_cells)

# 2. Evaluación !! Tipo 1 !! por Complejidad
evaluar_tipo1 = list(tipo1_cells - coincidencias)
# UMBRAL definido previamente (ej: 30.0)
tipo1_validadas = filtrar_sabs_por_complejidad(evaluar_tipo1, df_complejidad_sum, UMBRAL)

# Identificar flujos a eliminar (por pasar por celdas T1 complejas)
tipo1_descartadas = set(evaluar_tipo1) - set(tipo1_validadas)
flujos_a_eliminar = sab_tipo1[sab_tipo1['Cell_Name'].isin(tipo1_descartadas)]['Flujo'].unique()

# Filtrar SAB Tipo 1 final (usamos .copy() para evitar SettingWithCopyWarning)
sab1_final = sab_tipo1[~sab_tipo1['Flujo'].isin(flujos_a_eliminar)].copy()
flujo_localizados = sab1_final['Flujo'].unique()
celdas_t1_vivas = set(sab1_final['Cell_Name'])

# 3. Evaluación !! Tipo 2 !! (Conexión y Tamaño)
celdas_t2_puras = tipo2_cells - coincidencias
# Importante: .copy() aquí soluciona tu error de SettingWithCopyWarning
gdf_t2 = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_t2_puras)].copy()

# Unión de T1 validadas para detectar contacto
union_t1 = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_t1_vivas)].union_all()

# Clasificar T2 según contacto con T1
gdf_t2['toca_T1'] = gdf_t2.geometry.intersects(union_t1) 
t2_no_tocan_t1 = set(gdf_t2[~gdf_t2['toca_T1']]['Cell_Name'])
t2_que_tocan_t1 = set(gdf_t2[gdf_t2['toca_T1']]['Cell_Name'])

# 4. Análisis de Conectividad (Grafo Tipo 2)
G_t2 = nx.Graph()
G_t2.add_nodes_from(gdf_t2['Cell_Name']) # Aseguramos que todas las celdas existan en el grafo

nombres_t2 = gdf_t2['Cell_Name'].tolist()
geometrias_t2 = gdf_t2['geometry'].tolist()

for i in range(len(nombres_t2)):
    for j in range(i + 1, len(nombres_t2)):
        if geometrias_t2[i].intersects(geometrias_t2[j]):
            G_t2.add_edge(nombres_t2[i], nombres_t2[j])

# Filtrar grupos: si toca T1, requiere tamaño >= 3
t2_validadas_contacto = set()
for grupo in nx.connected_components(G_t2):
    if len(grupo) >= 3:
        # Validamos solo las celdas de este grupo que efectivamente tocan a T1
        validadas = grupo.intersection(t2_que_tocan_t1)
        t2_validadas_contacto.update(validadas)

# 5. Consolidación Final
tipo2_finales = t2_no_tocan_t1 | t2_validadas_contacto
sabs_finales_sistema = list(celdas_t1_vivas | tipo2_finales)

# Filtrar el DataFrame de grupos original
df_sabs2 = grupo_sab2[grupo_sab2['Cell_Name'].apply(lambda x: grupo_es_valido(x, tipo2_finales))].copy()
gdf_sab2_agrup = gpd.GeoDataFrame(df_sabs2, geometry='geometry')

# 5. Verificación
print(f"Total de grupos SAB analizados: {len(grupo_sab2['ID_SAB_Final'].unique())}")
print(f"Grupos SAB que contienen celdas Tipo 2 finales: {df_sabs2['ID_SAB_Final'].unique()}")

print(f"\n--- REPORTE TIPO 2 ---")
print(f"T2 que no tocan T1 (conservadas): {len(t2_no_tocan_t1)}")
print(f"T2 que tocan T1 evaluadas: {len(t2_que_tocan_t1)}")
print(f"T2 que tocan T1 y cumplen grupo >= 3: {len(t2_validadas_contacto)}")
# 6. RECALCULAR sabs_finales_sistema 
# Los SABs finales totales serían:
# (Los de Tipo 2) + (Los T1 evaluados que no eran complejos) + (Los T2 que conectan con los T1 pero tiene suficiente tamaño)
sabs_finales_sistema = list(celdas_t1_vivas) + list(tipo2_finales)

print('3')
## analizar los flujos para ver si son coincidentes
# 
dist_max = 0.3
ang_max = 20.0

geom_flujo_identf = []
grupo_f_ident = []

# for idx, flujo in DF_Flujos.iterrows():
#     flujo = flujo['Flujo_Clusterizado']
    
#     if flujo in flujo_localizados:
#         linea_flujo = flujo['Line']
#         geom_flujo_identf.append(flujo)
#         geom_flujo_identf.append(linea_flujo)
        
# for jdx, f_ident in geom_flujo_identf.iterrows():
#     f = f_ident['Flujo_Clusterizado']
#     f_line = f_ident['Line']
#     for idx, f_comp in geom_flujo_identf.iterrows():
#         f_c = f_comp['Flujo_Clusterizado']
#         f_comp_line = f_comp['Line']
#         cond = False

#         dist = f_line.distance(f_comp_line)
#         angulo = f_line.angle(f_comp_line)
        
#         cond = True if (dist<dist_max) and (angulo<ang_max)

#         if f_line.intersects(f_comp_line) or cond == True:
#             grupo_f = [f, f_c]
#             f_tag = f
#             grupo_f_ident.append(
#                 'Flujo represetante': f_tag,
#                 'Flujos identificados': grupo_f)

#             geom_flujo_identf.remove(f_ident)

print('4')
# --- FASE 1: Filtrado y extracción ---
for idx, row in DF_Flujos.iterrows():
    # Usamos 'row' para no sobrescribir la variable
    flujo_val = row['Flujo_Clusterizado']
    print(f'flujo a evaluar: {flujo_val}')

    for jdx in flujo_localizados:
        list_flujos = [grupo['Flujo_representante'] for grupo in grupo_f_ident]
        if flujo_val == jdx:
            print(f'flujo_val: {flujo_val}')
            # Guardamos como diccionarios dentro de una lista
            geom_flujo_identf.append({
                'Flujo_Clusterizado': flujo_val,
                'Line': row['Line']
            })

grupo_f_ident = []
procesados = set() # Usamos un 'set' para llevar control de los que ya hemos agrupado

# --- FASE 2: Agrupación Espacial ---
for i in range(len(geom_flujo_identf)):
    if i in procesados:
        continue  # Si ya fue agrupado con otra línea, lo saltamos
        
    f_ident = geom_flujo_identf[i]
    f_tag = f_ident['Flujo_Clusterizado']
    f_line = f_ident['Line']
    
    # Iniciamos el grupo con el flujo actual
    grupo_f = [f_tag]
    
    # Iteramos desde i+1 en adelante para no comparar consigo mismo 
    # y no repetir comparaciones previas (ej. si A se comparó con B, no comparamos B con A)
    for j in range(i + 1, len(geom_flujo_identf)):
        if j in procesados:
            continue
            
        f_comp = geom_flujo_identf[j]
        f_c = f_comp['Flujo_Clusterizado']
        f_comp_line = f_comp['Line']

        # Cálculo de métricas
        dist = f_line.distance(f_comp_line)
        angulo = f_line.angle(f_comp_line) 
        
        # Evaluamos la condición directamente
        cond = (dist < dist_max) and (angulo < ang_max)
        print(f'dist: {dist}')
        print(f'angulo: {angulo}')
        print(f'cond: {cond}')

        if f_line.intersects(f_comp_line) or cond:
            print(f'f_c: {f_c}')
            grupo_f.append(f_c)
            procesados.add(j) # Lo marcamos como procesado para sacarlo de futuras búsquedas base

    # Añadimos el resultado estructurado correctamente como diccionario
    grupo_f_ident.append({
        'Flujo_representante': f_tag,
        'Flujos_identificados': grupo_f
    })
    
    procesados.add(i)

list_flujos = []

list_flujos = [grupo['Flujo_representante'] for grupo in grupo_f_ident]

print(f'list_flujos: {list_flujos}')
print(f'flujos_localizados: {flujo_localizados}')






#%%
#-------------------------------------------------------------------------------- #
#------------------Diseño de las fronteras de sabs------------------------------- #
#-------------------------------------------------------------------------------- #

#%%
## Para los del tipo 1 -----------------------------------------------------------

# LOGICA: 
# 1. generar una línea paralela al flujo que se aleje del flujo con la misma distancia 
# que tiene el flujo con la frontera de referencia
# 2. extender esa línea para que corte el sector entero, y será la línea de corte para 
# dividir el sector en dos partes: el que contiene al flujo y el que no
# 3. La parte que contiene al flujo será el nuevo sab del tipo 1

print('1')
# Funcion para obtener la linea de corte
def rediseñar_frontera_tipo1(flujo_df, frontera_ref, sector_id):
    """
    Genera la nueva frontera paralela al flujo de Tipo 1.
    La línea se extiende hasta tocar los límites del sector original.
    """
    if flujo_df.empty:
        return None

    # 1. Definir la frontera exterior del sector (Boundary)
    frontera_original = frontera_ref
    
    # 2. Crear la LineString del flujo
    linea_flujo = flujo_df['Line'].iloc[0]
    print(f"Línea de flujo: {linea_flujo}")
   
    # 3. Calcular distancia flujo-frontera
    distancia = linea_flujo.distance(frontera_original)
    distancia = float(distancia)
    print(f"Distancia del flujo a la frontera: {distancia}")
    
    # 4. Generar offsets y elegir el que se aleja de la frontera actual
    off_r = linea_flujo.parallel_offset(distancia, side='right')
    off_l = linea_flujo.parallel_offset(-distancia, side='right')

    print(f"Offset derecha: {off_r}")
    print(f"Offset izquierda: {off_l}")
    
    distancia_r = off_r.distance(frontera_original)
    distancia_l = off_l.distance(frontera_original)
    
    if distancia_r > distancia_l:
        nueva_linea = off_r
    else:
        nueva_linea = off_l
    
    # 5. Extender la línea para que toque la frontera original (Corte de sección)
    # Creamos una línea muy larga en la misma dirección para asegurar la intersección
    factor_extension = 4  # Aumentar longitud para asegurar cruce
    coords = list(nueva_linea.coords)
    p1, p2 = coords[0], coords[-1]
    
    # Extensión lineal simple
    ext_p1 = (p1[0] + (p1[0] - p2[0]) * factor_extension, p1[1] + (p1[1] - p2[1]) * factor_extension)
    ext_p2 = (p2[0] + (p2[0] - p1[0]) * factor_extension, p2[1] + (p2[1] - p1[1]) * factor_extension)
    linea_extendida = LineString([ext_p1, ext_p2])
    
    # Recortamos la línea extendida con el polígono del sector
    sector_poly = DF_info_conf[DF_info_conf['SECTOR_ID'] == sector_id]['Contorno Sector'].iloc[0]
    nueva_frontera_final = linea_extendida.intersection(sector_poly)
    print(f"Nueva frontera final: {nueva_frontera_final}")
    return linea_flujo, nueva_frontera_final

# --- CÁLCULO DE NUEVOS SABs TIPO 1 (GEOMETRÍA DEFINITIVA) ---
sabs_tipo1_definitivos = []
nuevas_fronteras_t1 = [] # linea de corte
print('2')
# geometría de los sectores
gdf_acc = DF_info_conf.copy() 

for idx, flujo in DF_Flujos.iterrows():
    nombre_flujo = flujo['Flujo_Clusterizado']
    
    if nombre_flujo in flujo_localizados:
        linea_flujo = flujo['Line']
        
        # 1. IDENTIFICACIÓN ESPACIAL DEL SECTOR
        # Buscamos qué sector interseca con la línea del flujo
        sector_correspondiente = None
        for _, s_row in gdf_acc.iterrows():
            if s_row['Contorno Sector'].intersects(linea_flujo):
                sector_correspondiente = s_row
                break
        
        if sector_correspondiente is None:
            print(f"Aviso: El flujo {nombre_flujo} no interseca con ningún sector de la configuración.")
            continue
            
        sector_id_detectado = sector_correspondiente['SECTOR_ID']
        poly_sector = sector_correspondiente['Contorno Sector']
        
        # 2. Recuperar frontera de referencia para el rediseño (desde sab1_final)
        datos_sab1 = sab1_final[sab1_final['Flujo'] == nombre_flujo]
        if datos_sab1.empty: 
            continue
        frontera_ref = datos_sab1['Frontera_Interna'].iloc[0]
        
        # 3. Obtener la nueva frontera (línea de corte)
        # Usamos tu función rediseñar_frontera_tipo1
        resultado = rediseñar_frontera_tipo1(DF_Flujos[DF_Flujos['Flujo_Clusterizado'] == nombre_flujo], 
                                             frontera_ref, sector_id_detectado)
        
        if resultado:
            _, nueva_frontera_final = resultado
            nuevas_fronteras_t1.append((linea_flujo, nueva_frontera_final))
            # 4. Partir el sector original usando la nueva frontera extendida
            # Esto genera las piezas (el SAB y el resto del sector)
            
            factor_extension = 4  # Aumentar longitud para asegurar cruce
            coords = list(nueva_frontera_final.coords)
            p1, p2 = coords[0], coords[-1]
            
            # Extensión lineal simple
            ext_p1 = (p1[0] + (p1[0] - p2[0]) * factor_extension, p1[1] + (p1[1] - p2[1]) * factor_extension)
            ext_p2 = (p2[0] + (p2[0] - p1[0]) * factor_extension, p2[1] + (p2[1] - p1[1]) * factor_extension)
            f_extendida = LineString([ext_p1, ext_p2])
    
            piezas = split(poly_sector, f_extendida)
            
            print(f"las piezas son: {piezas}")
            
            # 5. Seleccionar la pieza que contiene el flujo
            sab_geom = None
            for pieza in piezas.geoms:
                if pieza.intersects(linea_flujo):
                    sab_geom = pieza
                    break
            
            if sab_geom:
                sabs_tipo1_definitivos.append({
                    'Flujo_ID': nombre_flujo,
                    'Sector_Detectado': sector_id_detectado,
                    'geometry': sab_geom,
                    'Tipo': 'SAB_Tipo1_Definitivo'
                })

print('3')
# --- CREACIÓN DEL DATAFRAME FINAL ---
df_sabs_t1_final = gpd.GeoDataFrame(sabs_tipo1_definitivos, geometry='geometry', crs=gdf_mallado.crs)

print(f"Se han generado {len(df_sabs_t1_final)} SABs definitivos de Tipo 1.")

# Visualización rápida para verificar
if not df_sabs_t1_final.empty:
    ax = df_sabs_t1_final.plot(color='cyan', alpha=0.5, edgecolor='blue', figsize=(10,10))
    plt.title("Geometrías Finales SABs Tipo 1")
    plt.show()

#%%
## Para los del tipo 2 -----------------------------------------------------------

#LOGICA:
# 1. Para cada grupo de tipo 2, se identifica el trozo de la frontera que engloba
# 2. Se busca los flujos cercanos a sab pero que no intersecan a la frontera que contiene
# Los flujos identificados pueden cortar la frontera, aunque no la parte de frontera 
# que contiene el sab tipo 2. Si cortan a alguna frontera, se divide el flujo en el 
# pt de corte.
# 3. Una vez obtenidos los flujos cercanos y sus posibles partes, se identifica a qué 
# sector pertenece cada uno, y si están partidos, también se identifica el sector.
# 4.Para cada sector, están los flujos pertenecientes, y se calcula la interpolación
# de las direcciones de los flujos para generar una línea que representa esos flujos. 
# Esa línea serán la línea de corte en ese sector, y cortará el sector formando partes 
# que serán los sabs

# Ya están las celdas del tipo 2 agrupados en el archivo 'grupo_sab2' y están filtrados 
# los grupos que conectan a los tipo 1 pero tiene un tamaño pequeño

#-----------------------------------------------------------------------------------
#%% Obtener las líneas de corte de cada sector para cada grupo de sab

sabs_tipo2_diseno_detallado = []
distancia_max_flujo = 0.5
factor_area_offset = 1  # Ajuste para la separación según el área del SAB
umbral_extremos= 0.01
tolerancia_paralelismo_grados = 30.0

# Previo análisis de flujos
def calcular_angulo(line):
    coords = np.array(line.coords)
    start, end = coords[0], coords[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    # Usamos módulo 180 para que dirección sea la misma sin importar el sentido
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return angle

def distancia_personalizada(idx1, idx2, dict_lineas, umbral_angular=10):
    """
    Calcula si dos líneas son 'vecinas' basándose en:
    1. Distancia mínima entre sus puntos de control.
    2. Diferencia angular estricta.
    """
    line1 = dict_lineas[int(idx1)]
    line2 = dict_lineas[int(idx2)]
    
    # 1. Filtro Angular Estricto
    ang1 = calcular_angulo(line1)
    ang2 = calcular_angulo(line2)
    diff_ang = abs(ang1 - ang2)
    diff_ang = min(diff_ang, 180 - diff_ang) # Circularidad
    
    if diff_ang > umbral_angular:
        return 999.0 # Infinito si no son paralelas

    # 2. Distancia Geométrica (Muestreamos puntos para comparar cercanía)
    # Comprobamos si algún punto de la línea corta está cerca de la larga
    pts1 = np.array(line1.coords)
    pts2 = np.array(line2.coords)
    
    # Distancia mínima entre cualquier punto de L1 y L2
    # Esto une líneas cortas con largas si están cerca
    dist_min = np.min([np.linalg.norm(p1 - p2) for p1 in pts1 for p2 in pts2])
    
    return dist_min

def generar_segmento_representativo(lista_lineas, longitud_segmento):
    """
    Crea un segmento de longitud fija orientado según la dirección media
    y centrado en el centroide geográfico del grupo.
    """
    if not lista_lineas: return None
    
    # 1. Calcular Centroide Medio (Ubicación)
    centroides = np.array([[l.centroid.x, l.centroid.y] for l in lista_lineas])
    cx, cy = np.mean(centroides, axis=0)
    
    # 2. Calcular Ángulo Medio (Dirección)
    # Importante: usamos el promedio de los componentes del vector para evitar 
    # problemas con ángulos que saltan de 179 a 0.
    vectores = []
    for l in lista_lineas:
        coords = np.array(l.coords)
        v = coords[-1] - coords[0]
        norm = np.linalg.norm(v)
        if norm > 0:
            uv = v / norm
            # Normalizar para que siempre apunte al mismo hemisferio (X positivo)
            if uv[0] < 0 or (uv[0] == 0 and uv[1] < 0):
                uv = -uv
            vectores.append(uv)
    
    v_medio = np.mean(vectores, axis=0)
    v_medio = v_medio / np.linalg.norm(v_medio) # Re-normalizar
    
    # 3. Construir los extremos del segmento desde el centroide
    # El segmento se extiende 'longitud_segmento/2' hacia cada lado
    p1 = [cx - v_medio[0] * (longitud_segmento / 2), 
          cy - v_medio[1] * (longitud_segmento / 2)]
    p2 = [cx + v_medio[0] * (longitud_segmento / 2), 
          cy + v_medio[1] * (longitud_segmento / 2)]
    
    return LineString([p1, p2])

def agrupar_por_sectores(df_flujos, df_sectores, dist_max_km, ang_max, min_flujos):
    """
    df_flujos: DataFrame con columna 'Line' (LineString)
    df_sectores: DataFrame con 'Contorno Sector' (Polygon) y 'SECTOR_ID'
    """
    resultados_totales = []
    lineas_medias_sectores = {} # Diccionario para guardar las backbone por sector

    # 1. ITERAR POR CADA SECTOR
    for _, sector_row in df_sectores.iterrows():
        sector_poly = sector_row['Contorno Sector']
        sector_id = sector_row['SECTOR_ID']
        flujos_en_sector = []

        print(f"Procesando Sector: {sector_id}...")
        
        # 2. RECORTAR FLUJOS (Intersección)
        # Solo nos quedamos con la parte del flujo que está dentro del sector
        flujos_en_sector = []
        for _, flujo_row in df_flujos.iterrows():
            if flujo_row['Line'].intersects(sector_poly):
                # Cortar la línea para que solo quede lo que está dentro del polígono
                fragmento = flujo_row['Line'].intersection(sector_poly)
                
                # Asegurarnos de que el resultado sea una LineString (a veces devuelve MultiLineString)
                if fragmento.geom_type == 'LineString':
                    flujos_en_sector.append({'Line': fragmento, 'Original_ID': flujo_row.name, 'Sector_ID': sector_id})
                elif fragmento.geom_type == 'MultiLineString':
                    for part in fragmento.geoms:
                        flujos_en_sector.append({'Line': part, 'Original_ID': flujo_row.name, 'Sector_ID': sector_id})

        if len(flujos_en_sector) < min_flujos:
            print(f"Sector {sector_id} ignorado por falta de flujos suficientes.")
            continue

        df_local = pd.DataFrame(flujos_en_sector)
        lineas_locales = df_local['Line'].tolist()
        n = len(lineas_locales)
        
        # 3. MATRIZ DE DISTANCIA LOCAL
        matriz_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = distancia_personalizada(i, j, lineas_locales, umbral_angular=ang_max)
                matriz_dist[i, j] = matriz_dist[j, i] = d

        # 4. CLUSTERING LOCAL
        db = DBSCAN(eps=dist_max_km, min_samples=min_flujos, metric='precomputed').fit(matriz_dist)
        df_local['cluster'] = db.labels_
        
        # Ajustar IDs de cluster para que sean únicos por sector (ej: "LECMAS_0")
        df_local['cluster_global'] = df_local['cluster'].apply(
            lambda x: f"{sector_id}_{x}" if x != -1 else -1
        )

        # 3. Cálculo de la línea representativa por cada grupo encontrado
        backbones_local = {}
        for c_id in df_local[df_local['cluster'] != -1]['cluster_global'].unique():
            subset_lines = df_local[df_local['cluster_global'] == c_id]['Line'].tolist()
            backbones_local[c_id] = generar_segmento_representativo(subset_lines, longitud_segmento=0.5)
        
        lineas_medias_sectores[sector_id] = backbones_local
        resultados_totales.append(df_local)

    # UNIFICAR RESULTADOS
    df_final = pd.concat(resultados_totales, ignore_index=True)

    # CREAR DATAFRAME DE ESPINAZOS (Para que el bucle posterior funcione)
    espinazos_list = []

    for s_id, backbones in lineas_medias_sectores.items():
        for c_id, line in backbones.items():
            # Contamos cuántos flujos hay en este cluster para el peso
            num_flujos = len(df_final[df_final['cluster_global'] == c_id])
            espinazos_list.append({
                'Sector_ID': s_id,
                'cluster_global': c_id,
                'Representative_Line': line,
                'Num_Flujos_Originales': num_flujos
            })
    
    df_espinazos = pd.DataFrame(espinazos_list)

    # # --- PLOT DE RESULTADOS POR SECTOR ---
    # plt.figure(figsize=(20, 15))
    # cmap = plt.get_cmap('tab20')
    
    # # Dibujar Sectores
    # for _, s_row in df_sectores.iterrows():
    #     x, y = s_row['Contorno Sector'].exterior.xy
    #     plt.fill(x, y, alpha=0.05, edgecolor='black', lw=1)
    #     plt.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
    #              s_row['SECTOR_ID'], fontsize=12, alpha=0.5, ha='center')

    # # Dibujar Flujos Agrupados
    # validos = df_final[df_final['cluster'] != -1]
    # clusters_unicos = validos['cluster_global'].unique()
    
    # for i, c_id in enumerate(clusters_unicos):
    #     subset = validos[validos['cluster_global'] == c_id]
    #     color = cmap(i % 20)
    #     sector_id = c_id.split('_')[0]

    #     # Dibujar flujos originales (finos)
    #     for _, row in subset.iterrows():
    #         plt.plot(*row['Line'].xy, color=color, lw=1, alpha=0.4)
        
    #     # Dibujar LÍNEA REPRESENTATIVA (Espinazo) - Discontinua y Gruesa
    #     linea_media = lineas_medias_sectores[sector_id].get(c_id)

    #     if linea_media:
    #         plt.plot(*linea_media.xy, color=color, lw=3, linestyle='--', label=f"Media {c_id}")
            
    #         # Etiqueta en el centroide de la línea media
    #         plt.text(linea_media.centroid.x, linea_media.centroid.y, f"{c_id.split('_')[1]}", 
    #                  fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))

    # plt.title(f"Flujos por Sector y sus Líneas Medias (Backbones)")
    # plt.tight_layout()
    # plt.show()
    # --- PLOT DE RESULTADOS POR SECTOR (ESCALA LOCAL) ---
    # fig, ax = plt.subplots(figsize=(20, 15)) # Tamaño grande para la vista global
    # cmap = plt.get_cmap('tab20')

    # # 1. Dibujar todos los Sectores (Fondo)
    # for _, s_row in df_sectores.iterrows():
    #     x, y = s_row['Contorno Sector'].exterior.xy
    #     ax.fill(x, y, alpha=0.05, edgecolor='black', lw=1.5)
    #     ax.text(s_row['Contorno Sector'].centroid.x, s_row['Contorno Sector'].centroid.y, 
    #             s_row['SECTOR_ID'], fontsize=12, alpha=0.4, ha='center', va='center', fontweight='bold')

    # # 2. Dibujar Flujos Agrupados y Espinazos
    # validos = df_final[df_final['cluster'] != -1]
    # clusters_unicos = validos['cluster_global'].unique()
    
    # for i, c_id in enumerate(clusters_unicos):
    #     subset = validos[validos['cluster_global'] == c_id]
    #     color = cmap(i % 20) # Se reciclan colores cada 20 clusters
    #     sector_id = c_id.split('_')[0]

    #     # Dibujar flujos originales (finos y translúcidos)
    #     for _, row in subset.iterrows():
    #         ax.plot(*row['Line'].xy, color=color, lw=1.5, alpha=0.3)
        
    #     # Dibujar LÍNEA REPRESENTATIVA (Backbone - Discontinua y Gruesa)
    #     linea_media = lineas_medias_sectores[sector_id].get(c_id)

    #     if linea_media:
    #         ax.plot(*linea_media.xy, color=color, lw=3.5, linestyle='--')
            
    #         # Etiqueta numérica sobre la línea media para identificarla en el mapa
    #         id_num = c_id.split('_')[1]
    #         ax.text(linea_media.centroid.x, linea_media.centroid.y, id_num, 
    #                 fontsize=10, fontweight='bold', 
    #                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=1.5))

    # # 3. Construir Leyenda Representativa Global
    # # Explicamos qué significa cada estilo de línea/polígono en lugar de listar 50 clusters
    # from matplotlib.patches import Patch

    # custom_lines = [
    #     Patch(facecolor='gray', alpha=0.1, edgecolor='black', lw=1.5, label='Sectores ACC'),
    #     Line2D([0], [0], color='gray', lw=1.5, alpha=0.5, label='Flujos Originales Agrupados'),
    #     Line2D([0], [0], color='black', lw=3.5, linestyle='--', label='Línea Representativa (Backbone)')
    # ]
    
    # ax.legend(handles=custom_lines, loc='upper right', title="Leyenda de Elementos", 
    #           fontsize=12, title_fontsize=14, framealpha=0.9)

    # # 4. Ajustes finales del gráfico
    # ax.set_title("Visión Global: Agrupación de Flujos y Líneas Medias por Sector", fontsize=18, fontweight='bold')
    # ax.set_xlabel("Longitud")
    # ax.set_ylabel("Latitud")
    # ax.grid(True, linestyle=':', alpha=0.6)
    
    # plt.tight_layout()
    # plt.show()
    
    return df_final, df_espinazos

# EJECUCIÓN
# DF_info_conf debe ser tu dataframe de sectores
df_resultado_sectores, df_espinazos_final = agrupar_por_sectores(DF_Flujos, DF_info_conf, dist_max_km=0.5, ang_max=5, min_flujos=5)
df_espinazos_final.to_csv(PATH_sabs + f'espinazos_flujos_{fecha_sel}.csv')

def plot_mapa_flujos_global(gdf_sectores, df_flujos_agrupados, df_espinazos):
    """
    Representa el mapa global de los sectores del ACC junto con los flujos 
    originales agrupados y sus líneas representativas (espinazos).
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = plt.get_cmap('tab20')

    # 1. DIBUJAR LOS SECTORES REALES (Fondo)
    # Asumimos que gdf_sectores es un GeoDataFrame válido
    gdf_sectores.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=1.5, alpha=0.6, zorder=1)
    
    for _, row in gdf_sectores.iterrows():
        centro = row.geometry.centroid
        ax.text(centro.x, centro.y, row['SECTOR_ID'], 
                fontsize=12, alpha=0.4, ha='center', va='center', fontweight='bold', zorder=2)

    # 2. DIBUJAR FLUJOS INDIVIDUALES AGRUPADOS
    # Filtramos el ruido (cluster == -1)
    validos = df_flujos_agrupados[df_flujos_agrupados['cluster'] != -1]
    clusters_unicos = validos['cluster_global'].unique()
    
    # Creamos un diccionario para asegurar que el espinazo y sus flujos tengan el mismo color
    color_dict = {c_id: cmap(i % 20) for i, c_id in enumerate(clusters_unicos)}

    for c_id in clusters_unicos:
        color = color_dict[c_id]
        subset = validos[validos['cluster_global'] == c_id]

        # Flujos base translúcidos
        for _, row in subset.iterrows():
            ax.plot(*row['Line'].xy, color=color, lw=1.5, alpha=0.25, zorder=3)

    # 3. DIBUJAR ESPINAZOS (Líneas representativas)
    for _, row in df_espinazos.iterrows():
        c_id = row['cluster_global']
        linea = row['Representative_Line']
        
        # Recuperamos el color asignado a este cluster
        color = color_dict.get(c_id, 'black')

        # Línea gruesa y discontinua
        ax.plot(*linea.xy, color=color, lw=3.5, linestyle='--', zorder=4)
        
        # Etiqueta numérica del cluster sobre el espinazo
        # (Extraemos el número después del ID del sector, ej: "LECMAS_2" -> "2")
        id_num = c_id.split('_')[1] if '_' in c_id else c_id
        ax.text(linea.centroid.x, linea.centroid.y, id_num,
                fontsize=10, fontweight='bold', zorder=5,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=1.5))

    # 4. LEYENDA REPRESENTATIVA
    # Usamos elementos neutros para no tener que listar los decenas de clusters
    custom_lines = [
        Line2D([0], [0], color='gray', lw=1.5, alpha=0.4, label='Flujos Originales'),
        Line2D([0], [0], color='black', lw=3.5, linestyle='--', label='Línea Media de grupo de flujos')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=12, title_fontsize=14)

    # 5. AJUSTES DEL GRÁFICO
    ax.set_title(f"Mapa de flujos agrupados por sectores\nFecha: {fecha_sel}-06-2022")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

plot_mapa_flujos_global(gdf_sectores, df_resultado_sectores, df_espinazos_final)


# Funciones para evaluar flujos cercanos
def crear_zona_contencion_simplificada(geom_celdas, factor_escala):
    """Crea un polígono simple (Convex Hull) de max 4-6 lados para limitar el diseño."""
    convex_hull = geom_celdas.convex_hull
    return shapely_scale(convex_hull, xfact=factor_escala, yfact=factor_escala, origin='centroid')

def es_zona_pasillo(zona_contencion, v_frontera, ratio_minimo_alargamiento=2.0, tolerancia_grados=30.0, plot=True):
    """
    Evalúa si una geometría tiene forma de 'pasillo' (corredor) alineado a un vector.

    Parámetros:
    - zona_contencion (Polygon): El polígono a evaluar.
    - v_frontera (np.array): Vector unitario de la dirección principal.
    - ratio_minimo_alargamiento (float): Relación mínima entre largo y ancho (L/W).
    - tolerancia_grados (float): Desviación máxima permitida respecto a la frontera.

    Retorna:
    - bool: True si es alargado y paralelo al vector, False en caso contrario.
    """
    # Controles de seguridad
    if zona_contencion is None or zona_contencion.is_empty or v_frontera is None:
        return False

    # 1. Sacamos el rectángulo mínimo rotado
    mrr = zona_contencion.minimum_rotated_rectangle
    coords_mrr = list(mrr.exterior.coords)
    
    # 2. Obtenemos los vectores de dos lados adyacentes del rectángulo
    p0, p1, p2 = np.array(coords_mrr[0]), np.array(coords_mrr[1]), np.array(coords_mrr[2])
    v1 = p1 - p0
    v2 = p2 - p1
    
    # 3. Calculamos la longitud de ambos lados
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    
    # Si por algún motivo matemático el polígono colapsa a una línea
    if l1 == 0 or l2 == 0:
        return False
        
    # Calculamos el alargamiento
    ratio_alargamiento = max(l1, l2) / min(l1, l2)
    
    # 4. Extraemos el vector direccional del lado MÁS LARGO
    v_largo = (v1 / l1) if l1 >= l2 else (v2 / l2)
    
    # 5. Calculamos el ángulo con el vector de la frontera
    dot_prod = np.dot(v_frontera, v_largo)
    # Usamos np.clip para evitar errores matemáticos de redondeo fuera de [-1, 1]
    angulo_rad = np.arccos(np.clip(abs(dot_prod), 0.0, 1.0))
    angulo_deg = np.degrees(angulo_rad)
    
    if ratio_alargamiento >= ratio_minimo_alargamiento and angulo_deg <= tolerancia_grados:
        es_pasillo = True
    else:
        es_pasillo = False
    

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Zona de contención (Morado claro)
        x, y = zona_contencion.exterior.xy
        ax.fill(x, y, alpha=0.3, color='mediumpurple', label='Zona Contención')
        ax.plot(x, y, color='purple', linewidth=1.5)

        # 2. Rectángulo Mínimo Rotado (MRR) (Línea discontinua negra)
        rx, ry = mrr.exterior.xy
        ax.plot(rx, ry, color='black', linestyle='--', linewidth=2, label='Rectángulo Rotado (MRR)')

        # Centroide para el origen de los vectores
        cx, cy = zona_contencion.centroid.x, zona_contencion.centroid.y
        
        # Escala visual para las flechas (40% de la longitud del lado más largo)
        escala = max(l1, l2) * 0.4 
        hw = escala * 0.08 # Ancho de la cabeza de flecha

        # 3. Dibujar v_largo (Azul)
        ax.arrow(cx, cy, v_largo[0]*escala, v_largo[1]*escala, 
                 head_width=hw, head_length=hw, fc='blue', ec='blue', 
                 linewidth=2.5, zorder=5, label='v_largo (Eje del Pasillo)')

        # 4. Dibujar v_frontera (Rojo)
        # Nota visual: Como evaluamos abs(dot_prod), la dirección real de v_frontera da igual.
        # Para que el gráfico sea intuitivo, dibujamos la flecha apuntando hacia el mismo lado que v_largo.
        v_front_grafico = v_frontera if dot_prod >= 0 else -v_frontera
        ax.arrow(cx, cy, v_front_grafico[0]*escala, v_front_grafico[1]*escala, 
                 head_width=hw, head_length=hw, fc='red', ec='red', 
                 linewidth=2.5, zorder=4, label='v_frontera (Referencia)')

        # Formato del gráfico
        color_titulo = 'green' if es_pasillo else 'red'
        ax.set_title(f"Resultado es_pasillo: {es_pasillo}\nRatio: {ratio_alargamiento:.2f} | Ángulo Desv: {angulo_deg:.1f}º", 
                     color=color_titulo, fontweight='bold')
        ax.set_aspect('equal') # Vital para que los ángulos se vean reales y no distorsionados
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Colocar la leyenda fuera del gráfico para que no tape las formas
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    # 6. Evaluación final
    return (ratio_alargamiento >= ratio_minimo_alargamiento) and (angulo_deg <= tolerancia_grados)


# Bucle para obtener las lineas de corte
for idx, grupo in df_sabs2.iterrows():
    geom_grupo = grupo['geometry']
    area_grupo = geom_grupo.area
    centro_grupo = geom_grupo.centroid
    
    # 1. identificar las fronteras que engloba el grupo de sab 
    # Identificar qué sectores toca este grupo de celdas
    sectores_adyacentes = gdf_sectores[gdf_sectores.intersects(geom_grupo.buffer(1e-7))]
    sectores = sectores_adyacentes['SECTOR_ID'].tolist()
    print(f"El grupo SAB {grupo['ID_SAB_Final']} toca los sectores: {sectores}")
    
    zona_contencion = crear_zona_contencion_simplificada(geom_grupo, 3)

    cortes_por_sector = []

    # Iterar por cada sector para definir su propia línea de corte paralela
    for sect in sectores:
        print(f"Procesando sector {sect} para el grupo SAB {grupo['ID_SAB_Final']}")
        
        s_id = sect
        s_poly = gdf_sectores[gdf_sectores['SECTOR_ID'] == s_id]['geometry'].iloc[0]
        f_interna= gdf_fronteras_internas[gdf_fronteras_internas['SECTOR_ID'] == s_id]

        # 1.1 Parte de frontera interna que engloba el grupo de sab
        f_estudio = MultiLineString(list(f_interna['geometry']))
        f_sab = f_estudio.intersection(geom_grupo.buffer(1e-7))
        v_frontera = None

        if not f_sab.is_empty:
            # Juntamos todas las coordenadas, sea línea simple o múltiple
            if f_sab.geom_type == 'LineString':
                coords_front = list(f_sab.coords)
            else: # MultiLineString
                coords_front = []
                for line in f_sab.geoms:
                    coords_front.extend(list(line.coords))
            
            if len(coords_front) >= 2:
                # Tomamos el primer punto
                pt_inicio = np.array(coords_front[0])
                # Calculamos las distancias a todos los demás puntos para encontrar el final real
                distancias = [np.linalg.norm(np.array(p) - pt_inicio) for p in coords_front]
                pt_fin = np.array(coords_front[np.argmax(distancias)]) # El punto más alejado
                
                u = pt_fin - pt_inicio
                norm_u = np.linalg.norm(u)
                if norm_u > 0:
                    v_frontera = u / norm_u # Vector unitario global y real de la frontera
        
        es_pasillo = es_zona_pasillo(zona_contencion, v_frontera)

        print(f"¿El sector {s_id} es un pasillo? {'Sí' if es_pasillo else 'No'}")

        if es_pasillo:
            estado_sab = 'Paralelo'
        else:
            estado_sab = 'Secante' 
        
        # 2. Búsqueda de flujos cercanos por sector
        vectores_sector = []
        flujos_identificados_sector = []
        pesos_longitud_sector = []
        lineas_identificadas = []
    
        # Agregamos el vector de la frontera para que influya en la dirección final, aunque no es un flujo real
        
        # Opción A: Analizar cada flujo
        # for _, f_row in DF_Flujos.iterrows():
        #     f_line = f_row['Line']

        #     dist = f_line.distance(geom_grupo)# Distancia general al grupo SAB
        #     dist_extremos = f_line.boundary.distance(f_sab) #Distancia específica de los EXTREMOS
        #     atraviesa_frontera = f_line.crosses(f_sab) #Evaluar si la línea cruza la frontera

        #     print(f"Distancia del flujo {f_row['Flujo_Clusterizado']} al grupo SAB: {dist}")

        #     # print(f"Distancia del flujo {f_row['Flujo_Clusterizado']} al grupo SAB: {dist}")

        #     # 1. EVALUACIÓN BASE (Aplica para pasillos y no pasillos)
        #     cumple_base = (dist < distancia_max_flujo) and (not atraviesa_frontera) and (dist_extremos > umbral_extremos)
            
        #     flujo_valido = False

        #     if cumple_base:
        #         if es_pasillo and v_frontera is not None:
        #             # --- CONDICIÓN EXTRA: Evaluar Paralelismo ---
        #             coords_f = list(f_line.coords)
        #             if len(coords_f) >= 2:
        #                 u_f = np.array(coords_f[-1]) - np.array(coords_f[0])
        #                 norm_u_f = np.linalg.norm(u_f)
                        
        #                 if norm_u_f > 0:
        #                     v_flujo = u_f / norm_u_f
        #                     dot_prod = np.dot(v_frontera, v_flujo)
        #                     angulo_flujo = np.degrees(np.arccos(np.clip(abs(dot_prod), 0.0, 1.0)))
        #                     print(f"Ángulo entre flujo {f_row['Flujo_Clusterizado']} y frontera: {angulo_flujo} grados")
        #                     # Es válido solo si viaja en paralelo a la frontera
        #                     if angulo_flujo <= tolerancia_paralelismo_grados:
        #                         flujo_valido = True
        #         else:
        #             # Si no es pasillo, ya es válido solo con cumplir las bases
        #             flujo_valido = True

        #     # CONDICIONES:
        #     # 1. dist < distancia_max_flujo (El flujo pasa cerca del SAB)
        #     # 2. not atraviesa_frontera (La línea no corta/atraviesa la frontera de lado a lado)
        #     # 3. dist_extremos > umbral_extremos (Los extremos no nacen ni mueren pegados a la frontera)        
            
        #     if flujo_valido:
        #         parte_interna = f_line.intersection(s_poly.buffer(1e-7))
        #     else:
        #         parte_interna = None

        #     if parte_interna is not None:            
        #         coords = list(parte_interna.coords) if parte_interna.geom_type == 'LineString' else list(parte_interna.geoms[0].coords)
        #         if len(coords) >= 2:
        #             v = np.array(coords[-1]) - np.array(coords[0])
        #             norm = np.linalg.norm(v)
        #             if norm > 0:
        #                 v_unit = v / norm   

        #                 vectores_sector.append(v_unit)
        #                 flujos_identificados_sector.append(f_line)

        #                 flujo_en_contencion = f_line.intersection(zona_contencion)
        #                 pesos_longitud_sector.append(flujo_en_contencion.length)

        # Opción B: Analizar cada línea representativa de los grupos de flujos
        # Asumimos que DF_Espinazos contiene las 'Representative_Line' calculadas previamente
        # y 'v_frontera' es el vector director de la frontera actual que quieres evaluar

        espinazos_del_sector = df_espinazos_final[df_espinazos_final['Sector_ID'] == s_id]

        for _, e_row in espinazos_del_sector.iterrows():
            e_line = e_row['Representative_Line']
            cluster_id = e_row['cluster_global']

            # 1. DISTANCIAS Y CRUCES (Usando la línea representativa)
            dist = e_line.distance(geom_grupo)  # Distancia al grupo SAB
            dist_extremos = e_line.boundary.distance(f_sab) # Distancia de los extremos a la frontera
            atraviesa_frontera = e_line.crosses(f_sab)

            # 1. EVALUACIÓN BASE (Aplica para pasillos y no pasillos)
            cumple_base = (dist < distancia_max_flujo) and (not atraviesa_frontera) and (dist_extremos > umbral_extremos)
            
            flujo_valido = False

            if cumple_base:
                if es_pasillo and v_frontera is not None:
                    print('Es pasillo y tiene frontera')
                    # --- CONDICIÓN EXTRA: Evaluar Paralelismo ---
                    coords_f = list(e_line.coords)
                    if len(coords_f) >= 2:
                        u_f = np.array(coords_f[-1]) - np.array(coords_f[0])
                        norm_u_f = np.linalg.norm(u_f)
                        
                        if norm_u_f > 0:
                            v_flujo = u_f / norm_u_f
                            dot_prod = np.dot(v_frontera, v_flujo)
                            angulo_flujo = np.degrees(np.arccos(np.clip(abs(dot_prod), 0.0, 1.0)))
                            print(f"Ángulo entre flujo {cluster_id} y frontera: {angulo_flujo} grados")
                            # Es válido solo si viaja en paralelo a la frontera
                            if angulo_flujo <= tolerancia_paralelismo_grados:
                                print(f'Línea {cluster_id} es válido')
                                flujo_valido = True
                else:
                    # Si no es pasillo, ya es válido solo con cumplir las bases
                    flujo_valido = True
                    print(f'Línea {cluster_id} es válido')

            # CONDICIONES:
            # 1. dist < distancia_max_flujo (El flujo pasa cerca del SAB)
            # 2. not atraviesa_frontera (La línea no corta/atraviesa la frontera de lado a lado)
            # 3. dist_extremos > umbral_extremos (Los extremos no nacen ni mueren pegados a la frontera)        
            
            # 2. EVALUACIÓN DE PARALELISMO

            if flujo_valido == True:
                # Extraemos el vector de la línea representativa (ya es un promedio, es muy fiable)
                coords_e = list(e_line.coords)
                u_e = np.array(coords_e[-1]) - np.array(coords_e[0])
                norm_u_e = np.linalg.norm(u_e)

                if norm_u_e > 0:
                    v_unit = u_e / norm_u_e
                    print(f"Línea Representativa {cluster_id} VÁLIDA para ajustar frontera.")
                    
                    # 4. PESO POR RELEVANCIA
                    # En vez de longitud simple, podemos pesar por el número de flujos que representa
                    peso = e_row['Num_Flujos_Originales'] * e_line.length
                    
                    vectores_sector.append(v_unit)
                    pesos_longitud_sector.append(peso)
                    lineas_identificadas.append(e_line)
   

         #4. interpolación de direcciones 
        # Si hay tráfico en este sector, generamos su línea de corte específica
        if vectores_sector:
            
            # # 1. Obtenemos las longitudes de los flujos para usarlas como pesos
            # pesos_longitud = [flujo.length for flujo in flujos_identificados_sector]
            
            # # 2. Calculamos la media ponderada de los vectores
            # v_medio_ponderado = np.average(vectores_sector, axis=0, weights=pesos_longitud)
            
            # # 3. Normalizamos el vector resultante para asegurar que siga siendo unitario
            # # (Vital para que el tamaño de la línea proyectada no se deforme)
            # norm_v = np.linalg.norm(v_medio_ponderado)
            # if norm_v > 0:
            #     v_medio = v_medio_ponderado / norm_v
            # else:
            #     v_medio = v_medio_ponderado # En el raro caso de que se cancelen exactamente
        
    
            # Ahora la suma de pesos suma 0 en el raro caso de que ningún flujo cruce la contención
            # suma_pesos = sum(pesos_longitud_sector)
            
            # if suma_pesos > 0:
            #     # Usamos la nueva lista de pesos que solo contiene la longitud recortada
            #     v_medio_ponderado = np.average(vectores_sector, axis=0, weights=pesos_longitud_sector)
            # else:
            #     # Fallback por si acaso todos intersectan en un solo punto (length 0)
            #     v_medio_ponderado = np.mean(vectores_sector, axis=0)
            
            # # Normalizamos el vector resultante
            # norm_v = np.linalg.norm(v_medio_ponderado)
            # if norm_v > 0:
            #     v_medio = v_medio_ponderado / norm_v
            # else:
            #     v_medio = v_medio_ponderado 
            

            # suma_pesos = sum(pesos_longitud_sector)
            # pesos = pesos_longitud_sector if suma_pesos > 0 else np.ones(len(vectores_sector))
            
            # # 1. Extraer los ángulos de todos los vectores (en radianes)
            # angulos = np.arctan2([v[1] for v in vectores_sector], [v[0] for v in vectores_sector])
            
            # # 2. Multiplicar por 2 (Convierte ejes bidireccionales en unidireccionales)
            # # Ej: 10º y 190º se convierten en 20º y 380º (que equivale a 20º)
            # angulos_dobles = angulos * 2
            
            # # 3. Calcular la media ponderada de los componentes trigonométricos
            # mean_cos = np.average(np.cos(angulos_dobles))
            # mean_sin = np.average(np.sin(angulos_dobles))
            
            # # 4. Obtener el ángulo medio resultante y dividirlo por 2 para volver a la escala real
            # angulo_medio_final = np.arctan2(mean_sin, mean_cos) / 2.0
            
            # # 5. Reconstruir el vector medio final (un vector unitario perfecto)
            # v_medio = np.array([np.cos(angulo_medio_final), np.sin(angulo_medio_final)])
            

            # 1. Crear matriz de dispersión 2x2
            S = np.zeros((2, 2))
            
            for v in vectores_sector:
                # Aseguramos que v es un array columna de 2x1
                v_col = np.array(v).reshape(2, 1)
                
                # Multiplicar el vector por su traspuesta da una matriz direccional.
                # ¡Magia!: v_col @ v_col.T da exactamente lo mismo si el avión va de ida o de vuelta.
                S += v_col @ v_col.T 
            
            # 2. Calcular los Autovalores (Eigenvalues) y Autovectores (Eigenvectors)
            # np.linalg.eigh es específico y muy rápido para matrices simétricas como S
            eigenvalues, eigenvectors = np.linalg.eigh(S)
            
            # 3. El eje principal (nuestra línea azul) es el autovector con el autovalor más alto
            v_medio = eigenvectors[:, np.argmax(eigenvalues)]

            # angulo_f = np.arctan2(v_medio_f[1], v_medio_f[0])
            # angulo_front = np.arctan2(v_frontera[1], v_frontera[0])
            
            # angulos_fd = angulo_f * 2
            # angulos_front_d = angulo_front * 2
            # # 3. Calcular la media ponderada de los componentes trigonométricos
            # mean_cos = np.average(np.cos([angulos_fd, angulos_front_d]))
            # mean_sin = np.average(np.sin([angulos_fd, angulos_front_d]))
            
            # # 4. Obtener el ángulo medio resultante y dividirlo por 2 para volver a la escala real
            # angulo_medio_final = np.arctan2(mean_sin, mean_cos) / 2.0
            
            # # 5. Reconstruir el vector medio final (un vector unitario perfecto)
            # v_medio = np.array([np.cos(angulo_medio_final), np.sin(angulo_medio_final)])

            # Cálculo del offset y la línea de corte (igual que antes)
            offset_dist = np.sqrt(area_grupo) * factor_area_offset
            
            p1 = (centro_grupo.x - v_medio[0]*5, centro_grupo.y - v_medio[1]*5)
            p2 = (centro_grupo.x + v_medio[0]*5, centro_grupo.y + v_medio[1]*5)
            linea_corte = LineString([p1, p2])

            cortes_por_sector.append({
                'Sector_ID': s_id,
                'Direccion_Media': v_medio,
                'Linea_Corte': linea_corte,
                'Offset': offset_dist,
                'Lineas_Identificadas': lineas_identificadas
            })        
    if cortes_por_sector:
        sabs_tipo2_diseno_detallado.append({
            'ID_SAB_Final': grupo['ID_SAB_Final'],
            'Geom_Grupo': geom_grupo,
            'Area': area_grupo,
            'Disenos_Sectoriales': cortes_por_sector,
            'Celdas': grupo['Cell_Name'],
            'Frontera intera': f_sab,
            'Estado de sab': estado_sab
        })    

#----------------------------------------------------------------------------------
# Ejecución:
df_referencia_t2_final = pd.DataFrame(sabs_tipo2_diseno_detallado)
df_referencia_t2_final.to_pickle(PATH_sabs + f'diseno_detallado_sab2_{fecha_sel}.pkl')
df_referencia_t2_final.to_csv(PATH_sabs + f'diseno_detallado_sab2_{fecha_sel}.csv', index=False)

print(f"Generados diseños sectoriales para {len(df_referencia_t2_final)} grupos SAB Tipo 2.")

# Visualización:
def plot_flujos_seleccionados(gdf_sectores, df_flujos, df_referencia, sample_id=None):
    if sample_id is None:
        if df_referencia.empty:
            return
        sample_id = df_referencia.iloc[0]['ID_SAB_Final']

    sab_data = df_referencia[df_referencia['ID_SAB_Final'] == sample_id]
    if sab_data.empty:
        return
    
    sab_row = sab_data.iloc[0]
    sectores_involucrados = [d['Sector_ID'] for d in sab_row['Disenos_Sectoriales']]
    gdf_sectores_zoom = gdf_sectores[gdf_sectores['SECTOR_ID'].isin(sectores_involucrados)]
    # zona_contencion = crear_zona_contencion_simplificada(sab_row['Geom_Grupo'], 3)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Fondos
    gdf_sectores_zoom.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    gdf_flujos_totales = gpd.GeoDataFrame(df_flujos, geometry='Representative_Line')
    gdf_flujos_totales.plot(ax=ax, color='lightgray', linewidth=1, alpha=0.3, zorder=2)
    # gpd.GeoSeries([zona_contencion]).plot(ax=ax, facecolor='mediumpurple', edgecolor='purple', linewidth=2, linestyle='-.', alpha=0.15, zorder=3)
    gpd.GeoSeries([sab_row['Geom_Grupo']]).plot(ax=ax, color='orange', alpha=0.4, edgecolor='darkorange', zorder=4)

    # --- LA MAGIA: PALETA DE COLORES POR SECTOR ---
    colores = ['red', 'green', 'blue', 'magenta', 'cyan', 'darkorange']
    custom_lines = [
        # Line2D([0], [0], color='lightgray', lw=2, label='Flujos Descartados'),
        mpatches.Patch(color='orange', alpha=0.4, label='Celdas del tipo 2')
        # mpatches.Patch(facecolor='mediumpurple', edgecolor='purple', linestyle='-.', alpha=0.3, label='Zona Contención')
    ]

    # Iterar asignando un color a cada sector
    for i, d in enumerate(sab_row['Disenos_Sectoriales']):
        color_sector = colores[i % len(colores)]
        s_id = d['Sector_ID']
        
        flujos_validos = d.get('Lineas_Identificadas', [])
        if flujos_validos:
            # Pintamos los flujos de este sector de su color
            gpd.GeoSeries(flujos_validos).plot(ax=ax, color=color_sector, linewidth=2.5, alpha=0.8, zorder=5)
        
        # Pintamos la línea de corte del MISMO color
        l_corte = d['Linea_Corte']
        x, y = l_corte.xy
        mitad = len(x) // 2
        
        ax.plot(x, y, color=color_sector, linewidth=2, linestyle='--', zorder=6)
        ax.text(x[mitad], y[mitad], s_id, color=color_sector, fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_sector, pad=1.5))
        
        # Añadimos a la leyenda
        custom_lines.append(Line2D([0], [0], color=color_sector, lw=3,linestyle='--', label=f'Flujos identificados para sector {s_id}'))

    ax.set_title(f"Evaluación de Flujos para celdas del tipo 2\nFecha: {fecha_sel}-06-2022", fontsize=14)
    ax.legend(handles=custom_lines, loc='lower right')
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

# --- EJECUCIÓN ---
plot_flujos_seleccionados(gdf_sectores, df_espinazos_final, df_referencia_t2_final)
plot_flujos_seleccionados(gdf_sectores, df_espinazos_final, df_referencia_t2_final, sample_id=df_referencia_t2_final['ID_SAB_Final'].iloc[1])

#%% Recorte de los sectores

# 
def extender_linea_pro(linea, factor):
    """Extiende la línea drásticamente para asegurar colisiones con sectores."""
    coords = list(linea.coords)
    p1, p2 = np.array(coords[0]), np.array(coords[1])
    vector = p2 - p1
    return LineString([p1 - vector * factor, p2 + vector * factor])

def orientar_vector_hacia_centroide(v_normal, linea_base, geom_sab):
    """
    Evalúa si un vector normal apunta hacia el centroide de una geometría.
    Si apunta en dirección contraria (hacia afuera), lo invierte.
    
    Parámetros:
        v_normal (np.array): Vector perpendicular al flujo [x, y].
        linea_base (LineString): Línea de corte desde donde se proyecta el vector.
        geom_sab (Polygon): Geometría del SAB para calcular su centroide.
        
    Retorna:
        np.array: El vector normal garantizado para apuntar hacia el centroide.
    """
    # 1. Tomar un punto de referencia en la línea (primer vértice)
    punto_en_linea = np.array([linea_base.coords[0][0], linea_base.coords[0][1]])
    
    # 2. Obtener el centroide del grupo SAB
    centroide = np.array([geom_sab.centroid.x, geom_sab.centroid.y])
    
    # 3. Calcular el vector direccional puro hacia el centroide
    vector_hacia_centroide = centroide - punto_en_linea
    
    # 4. Evaluar con el producto punto
    producto_punto = np.dot(v_normal, vector_hacia_centroide)
    
    if producto_punto < 0:
        # Se aleja del centroide, lo invertimos
        return -v_normal
    else:
        # Ya apunta hacia el centroide (o pasa por encima)
        return v_normal


# def optimizar_offset_por_area(poly_sector_real, centroide_sector, geom_original, l_corte_base, area_objetivo, v_medio):
#     """
#     Búsqueda binaria para encontrar el offset que resulte en el área exacta.
#     """
#     v_normal_inicial = np.array([-v_medio[1], v_medio[0]]) # Perpendicular al flujo
#     v_normal = orientar_vector_hacia_centroide(v_normal_inicial, l_corte_base, geom_original)

#     # Rango de búsqueda del offset (en grados/coordenadas)
#     d_min = -0.5  # Hacia el interior del SAB
#     d_max = 0.5   # Alejándose del SAB
#     best_geom = None
#     best_offset = 0
    
#     for _ in range(12):  # 12 iteraciones dan una precisión de < 0.1%
#         d_mid = (d_min + d_max) / 2
#         # Desplazamos la línea
#         l_probada = translate(l_corte_base, xoff=v_normal[0]*d_mid, yoff=v_normal[1]*d_mid)
        
#         try:
#             partes = split(poly_sector_real, l_probada)
#             candidato = None
#             for p in partes.geoms:
#                 if p.intersects(geom_original):
#                     candidato = p
#                     break
            
#             if candidato is None:
#                 d_max = d_mid # Demasiado lejos
#                 continue
                
#             if candidato.area > area_objetivo:
#                 # El polígono es muy grande, mover línea para "achicar" (alejar del centroide)
#                 d_min = d_mid
#                 best_offset = d_mid
#                 best_geom = candidato
#             else:
#                 # El polígono es muy pequeño, mover línea hacia el centroide
#                 d_max = d_mid
#         except:
#             d_max = d_mid
            
#     return best_geom, best_offset

# # --- BUCLE PRINCIPAL DE REDISEÑO TIPO 2 ---

# sabs_diseno_finalizado = []

# for idx, sab in df_referencia_t2_final.iterrows():
#     geom_original_celdas = sab['Geom_Grupo']
#     cortes = sab['Disenos_Sectoriales']
#     s_id_sab = sab['ID_SAB_Final']
    
#     # Requisito: Polígono de contención simple (Convex Hull)
#     zona_limite = crear_zona_contencion_simplificada(geom_original_celdas, factor_escala=2.0)
    
#     area_por_sector = sab['Area'] / len(cortes)
#     geometrias_sectores = []
    
#     for d in cortes:
#         # Obtener sector real
#         poly_sector_real = gdf_sectores.loc[gdf_sectores['SECTOR_ID'] == d['Sector_ID'], 'geometry'].values[0]
#         centroide_sector = poly_sector_real.centroid
#         # Línea de flujo base extendida
#         l_base = extender_linea_pro(d['Linea_Corte'], factor=100)
        
#         # Optimizar posición de la línea de corte basada en área
#         geom_optima, offset_calc = optimizar_offset_por_area(
#             poly_sector_real, 
#             centroide_sector,
#             geom_original_celdas, 
#             l_base, 
#             area_por_sector, 
#             d['Direccion_Media']
#         )
        
#         if geom_optima:
#             geometrias_sectores.append(geom_optima)

#     if geometrias_sectores:
#         # Unir piezas y asegurar que no excedan la zona de contención ni el sector real
#         sab_final_geom = unary_union(geometrias_sectores)
#         sab_final_geom = sab_final_geom.intersection(zona_limite)
        
#         sabs_diseno_finalizado.append({
#             'ID_SAB_Final': s_id_sab,
#             'Geometry_Detallada': sab_final_geom,
#             'Area_Final': sab_final_geom.area,
#             'Sectores': [d['Sector_ID'] for d in cortes]
#         })

def obtener_pieza_sab_t2_paral(poly_sector, l_corte, geom_original_celdas):
    """
    Divide el sector y devuelve la pieza que representa el SAB.
    """
    # 1. Asegurar que la línea de corte sea lo bastante larga para cruzar el sector
    l_extendida = extender_linea_pro(l_corte, factor=50)
    zona_contencion = crear_zona_contencion_simplificada(geom_original_celdas, factor_escala=2.0)
    frontera_contenida = poly_sector.intersection(geom_original_celdas)

    try:
        
        piezas = split(poly_sector, l_extendida)
        
        # El SAB es la pieza que contiene/toca las celdas originales
        for pieza in piezas.geoms:
            # Usamos un pequeño buffer para asegurar la intersección 
            if pieza.intersects(geom_original_celdas.buffer(1e-6)):
                return pieza
    except Exception as e:
        print(f"Error en split: {e}")
        return None
    return None


def obtener_pieza_sab_t2_sec(poly_sector, l_corte, geom_original_celdas):
    """
    Divide el sector y devuelve la pieza que representa el SAB.
    Si la línea de flujo no corta la frontera, se usa la línea entre extremos de la frontera.
    """
    # 1. Definir la frontera interna (donde están las celdas)
    # Usamos buffer(0) para limpiar posibles errores topológicos
    frontera_contenida = poly_sector.boundary.intersection(geom_original_celdas.buffer(1e-7))
    
    l_extendida = extender_linea_pro(l_corte, factor=50)

    try:
        # Evaluamos si la línea de corte calculada por flujos corta la frontera interna
        if l_extendida.intersects(frontera_contenida) and not l_extendida.crosses(frontera_contenida):
            # CASO A: La línea de flujo es válida para el corte
            print('caso A:La línea de flujo es válida')
            linea_para_cortar = l_extendida
        else:
            # CASO B: La línea no intersecta o es secante de forma inválida.
            # Forzamos la línea recta entre los extremos de la frontera contenida.
            print('caso B:La línea no intersecta o es secante de forma inválida')
            if not frontera_contenida.is_empty:
                # Obtenemos los puntos extremos de la frontera (inicio y fin)
                coords_f = []
                if frontera_contenida.geom_type == 'LineString':
                    coords_f = [frontera_contenida.coords[0], frontera_contenida.coords[-1]]
                elif frontera_contenida.geom_type in ['MultiLineString', 'GeometryCollection']:
                    # Si es compleja, tomamos los puntos más alejados entre sí
                    all_coords = []
                    for geom in frontera_contenida.geoms:
                        if hasattr(geom, 'coords'): all_coords.extend(list(geom.coords))
                    if all_coords:
                        p_start = all_coords[0]
                        # Buscamos el punto más lejano al inicial
                        p_end = max(all_coords, key=lambda p: np.linalg.norm(np.array(p)-np.array(p_start)))
                        coords_f = [p_start, p_end]
                
                if len(coords_f) >= 2:
                    l_corte_mod = LineString([coords_f[0], coords_f[1]])
                    # IMPORTANTE: Extenderla para que el split no falle por precisión
                    linea_para_cortar = extender_linea_pro(l_corte_mod, factor=10)
                else:
                    return None
            else:
                return None

        # 2. Ejecutar el corte
        piezas = split(poly_sector, linea_para_cortar)
        
        # 3. Selección de la pieza (Recinto del SAB)
        for pieza in piezas.geoms:
            # La pieza correcta es la que contiene las celdas originales
            if pieza.intersects(geom_original_celdas.buffer(1e-6)):
                print('Pieza encontrada')
                return pieza
                
    except Exception as e:
        print(f"Error en split T2: {e}")
        return None
        
    return None

def optimizar_diseno_t2_paral(poly_sector, geom_original_celdas, l_corte_base, v_medio, area_objetivo, f_sab):
    """
    Busca el offset ideal para que el recinto entre frontera y línea cumpla el área.
    """
    # Definir dirección de movimiento (Normal a la línea de flujo)
    v_normal_inicial = np.array([-v_medio[1], v_medio[0]])
    v_normal = orientar_vector_hacia_centroide(v_normal_inicial, l_corte_base, geom_original_celdas)

    d_min = 0.0  # Hacia adentro
    d_max = 1   # Hacia afuera
    
    mejor_pieza = None
    
    # Búsqueda iterativa (Binaria)
    for i in range(30):
        d_mid = (d_min + d_max) / 2
        l_probada = translate(l_corte_base, xoff=v_normal[0]*d_mid, yoff=v_normal[1]*d_mid)
        l_corte = extender_linea_pro(l_probada, 5)
        
        piezas = split(poly_sector, l_corte)
        
        for pieza in piezas.geoms:
            if pieza.intersects(f_sab):
                pieza_candidata = pieza
                break
        
        if pieza_candidata is None:
            d_max = d_mid # Si no hay pieza, estamos fuera del sector
            continue
            
        area_actual = pieza_candidata.area
        
        # Criterio de parada (si estamos a menos del 2% del área objetivo)
        if abs(area_actual - area_objetivo) / area_objetivo < 0.02:
            mejor_pieza = pieza_candidata
            break
           
        if area_actual < area_objetivo:
            # Muy pequeño: alejar la línea de la frontera (hacia afuera)
            # Dependiendo de la orientación, esto suele ser aumentar d
            d_min = d_mid
        else:
            # Muy grande: acercar la línea a la frontera
            d_max = d_mid
    
    if mejor_pieza:
        print('SAB paralelo: Pieza encontrada') 
    else:
        print('SAB paralelo: Pieza no encontrada')     


    return mejor_pieza

def optimizar_diseno_t2_sec(lista_sectores_data, f_sab, area_objetivo, poly_total_limite):
    """
    Optimiza la posición de las líneas de corte a lo largo de f_sab para 
    alcanzar un área total combinada.
    
    Args:
        lista_sectores_data: Lista de dicts con 'Sector_ID', 'Direccion_Media', etc.
        f_sab: LineString que define la frontera de deslizamiento.
        area_objetivo: El área total que debe sumar la pieza compuesta.
        poly_total_limite: El polígono que engloba todos los sectores (para el split).
    """
    
    # Rango de búsqueda: parámetro t de f_sab (de 0 a 1, normalizado)
    # Se puede extender un poco (e.g., -0.1 a 1.1) si f_sab es corta
    t_min = 0.0
    t_max = 1.0
    
    mejor_union = None
    
    for i in range(30):  # Búsqueda binaria
        t_mid = (t_min + t_max) / 2
        punto_ancla = f_sab.interpolate(t_mid, normalized=True)
        
        piezas_iteracion = []
        
        for sector in lista_sectores_data:
            poly_sector = sector['poly_sector']
            v_medio = sector['Direccion_Media']
            l_base = sector['Linea_Corte']
            
            # 1. Mover la línea de corte para que pase por el punto_ancla de la frontera
            # Calculamos cuánto hay que desplazar el centro de la línea original al punto de la frontera
            centro_linea = l_base.interpolate(0.5, normalized=True)
            dx = punto_ancla.x - centro_linea.x
            dy = punto_ancla.y - centro_linea.y
            
            l_probada = translate(l_base, xoff=dx, yoff=dy)
            l_corte = extender_linea_pro(l_probada, 5)
            
            # 2. Cortar el sector
            resultado_split = split(poly_sector, l_corte)
            
            # 3. Identificar la pieza correcta (la que contiene las líneas identificadas o celdas)
            pieza_sector = None
            min_area = float('inf') # Empezamos con un valor infinito
        
            for pieza in resultado_split.geoms:
                # Verificamos que sea una geometría válida y comparamos áreas
                if pieza and not pieza.is_empty:
                    area_p = pieza.area
                    if area_p < min_area:
                        min_area = area_p
                        pieza_sector = pieza
            
            if pieza_sector:
                piezas_iteracion.append(pieza_sector)
        
        # 4. Evaluar el área total de la pieza compuesta
        if not piezas_iteracion:
            t_min = t_mid # Ajustar según lógica de dirección
            continue
            
        union_actual = unary_union(piezas_iteracion)
        area_actual = union_actual.area
        print(f'area de pieza es {area_actual} y objetivo es {area_objetivo}')
        # 5. Criterio de parada (2% de error)
        if abs(area_actual - area_objetivo) / area_objetivo < 0.5:
            
            mejor_union = union_actual
            break

        # mejor_union = union_actual
        # 6. Ajuste de búsqueda binaria
        # Nota: Dependiendo de si aumentar 't' agranda o achica el área, 
        # puede que necesites invertir estos signos.
        if area_actual > area_objetivo:
            t_min = t_mid
        else:
            t_max = t_mid
    
    if mejor_union:
        print('SAB secante: Pieza encontrada') 
    else:
        print('SAB secante: Pieza no encontrada')     

    return mejor_union


sabs_diseno_finalizado = []

for idx, sab in df_referencia_t2_final.iterrows():
    geom_celdas = sab['Geom_Grupo']
    cortes = sab['Disenos_Sectoriales']
    s_id_sab = sab['ID_SAB_Final']
    estado_sab = sab['Estado de sab']
    f_sab = sab['Frontera intera']

    print(f'frontera de sab: {f_sab}')

    # El área objetivo de cada "trozo" de sector
    # Generalmente es el área de las celdas que caen en ese sector
    area_total_sab = geom_celdas.area
    num_sectores = len(cortes)
    area_target_por_pieza = area_total_sab/ num_sectores*1.25

    piezas_finales_del_sab = []

    if estado_sab == 'Paralelo':
        for d in cortes:
            poly_sector = gdf_sectores.loc[gdf_sectores['SECTOR_ID'] == d['Sector_ID'], 'geometry'].values[0]
            
            # Línea de corte calculada por los flujos
            l_corte_input = d['Linea_Corte']
            v_dir = d['Direccion_Media']

            print(f'Estado de sab = {estado_sab}')
            # Ejecutar optimización para este sector
            pieza_optima = optimizar_diseno_t2_paral(
                poly_sector, 
                geom_celdas, 
                l_corte_input, 
                v_dir, 
                area_target_por_pieza,
                f_sab
                )
            if pieza_optima:
                piezas_finales_del_sab.append(pieza_optima)

            if piezas_finales_del_sab:
                geom_final_unida = unary_union(piezas_finales_del_sab)
    else:

        for sector_info in cortes:
            s_id = sector_info['Sector_ID']
            # Buscamos el polígono en el GeoDataFrame original de sectores
            poly_geo = gdf_sectores.loc[gdf_sectores['SECTOR_ID'] == s_id, 'geometry'].values[0]
            sector_info['poly_sector'] = poly_geo

        pieza_optima = optimizar_diseno_t2_sec(
            cortes,
            f_sab, 
            area_total_sab,
            geom_celdas
            )
        
        if pieza_optima:
            geom_final_unida=pieza_optima
            print(f'geom final = {geom_final_unida}')

    if geom_final_unida:
        sabs_diseno_finalizado.append({
            'ID_SAB_Final': s_id_sab,
            'Geometry_Detallada': geom_final_unida,
            'Area_Final': geom_final_unida.area,
            'Sectores_Afectados': [d['Sector_ID'] for d in cortes]
        })

gdf_sabs_tipo2_detallado = gpd.GeoDataFrame(sabs_diseno_finalizado, geometry='Geometry_Detallada', crs=gdf_sectores.crs)
gdf_sabs_tipo2_detallado.to_csv(PATH_sabs + f'gdf_sab2_final_{fecha_sel}.csv')

def plot_final_optimizacion(gdf_sectores, gdf_celdas_orig, gdf_final, df_referencia, sample_id):
    # Seleccionar SAB
    if sample_id is None: sample_id = gdf_final.iloc[1]['ID_SAB_Final']
    
    sab_data = gdf_final[gdf_final['ID_SAB_Final'] == sample_id].iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Sectores Reales
    gdf_sectores[gdf_sectores['SECTOR_ID'].isin(sab_data['Sectores_Afectados'])].plot(ax=ax, color='white', edgecolor='black', lw=1.5, alpha=0.3)
    
    # 2. Líneas de Corte de la Referencia (Flujos)
    disenos = df_referencia[df_referencia['ID_SAB_Final'] == sample_id]['Disenos_Sectoriales'].iloc[0]
    for d in disenos:
        l_ext = extender_linea_pro(d['Linea_Corte'], factor=0.02)
        s_id = d['Sector_ID']
        x, y = l_ext.xy
        mitad = len(x) // 2
        ax.plot(*l_ext.xy, color='orange', ls='--', label='Eje de Flujo')
        ax.text(x[mitad], y[mitad], s_id, 
                    color='darkred', fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))


    # 3. Celdas Originales
    geom_celdas = gdf_celdas_orig[gdf_celdas_orig['ID_SAB_Final'] == sample_id].geometry.iloc[0]
    gpd.GeoSeries([geom_celdas]).plot(ax=ax, color='blue', alpha=0.1, label='Celdas Originales')
    
    # 4. Resultado FINAL Optimizado
    gpd.GeoSeries([sab_data['Geometry_Detallada']]).plot(ax=ax, color='red', alpha=0.6, edgecolor='darkred', lw=2)

    ax.set_title(f"SAB {sample_id}: Optimización por Área y Frontera Real")
    plt.show()

# Ejecutar validación
plot_final_optimizacion(gdf_sectores, df_sabs2, gdf_sabs_tipo2_detallado, df_referencia_t2_final, sample_id=df_referencia_t2_final.iloc[0]['ID_SAB_Final'])

gdf_sabs_tipo2_detallado = gpd.GeoDataFrame(sabs_diseno_finalizado, geometry='Geometry_Detallada', crs=gdf_sectores.crs)
df_final_geom = pd.DataFrame(sabs_diseno_finalizado)

# --- EJECUCIÓN ---
plot_final_optimizacion(
    gdf_sectores, 
    df_sabs2, 
    gdf_sabs_tipo2_detallado, 
    df_referencia_t2_final, 
    sample_id=df_referencia_t2_final.iloc[0]['ID_SAB_Final']
)
plot_final_optimizacion(
    gdf_sectores, 
    df_sabs2, 
    gdf_sabs_tipo2_detallado, 
    df_referencia_t2_final, 
    sample_id=df_referencia_t2_final.iloc[1]['ID_SAB_Final']
)


print(f"Diseño detallado finalizado para {len(gdf_sabs_tipo2_detallado)} SABs Tipo 2.")

# --- REPRESENTACIÓN GRÁFICA TIPO 2: CORTES MULTI-SECTORIALES ---
fig, ax7 = plt.subplots(figsize=(12, 10))

# 1. Fondo del ACC
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    ax7.fill(x, y, alpha=0.1, edgecolor='black', linewidth=1, label='_nolegend_')
    ax7.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], fontsize=8, alpha=0.4)

# 2. Grupos SAB Tipo 2
if not df_sabs2.empty:
    gpd.GeoDataFrame(df_sabs2, geometry='geometry').plot(ax=ax7, color='orange', alpha=0.3, label='Grupos SAB 2')

# 3. Dibujar cortes por cada sector afectado
for idx, row in df_referencia_t2_final.iterrows():
    for d_sect in row['Disenos_Sectoriales']:
        l_corte = d_sect['Linea_Corte']
        s_id = d_sect['Sector_ID']
        # Dibujamos la línea de corte punteada
        x_c, y_c = l_corte.xy
        mitad = len(x_c) // 2
        ax7.plot(x_c, y_c, color='blue', linewidth=1.2, linestyle='--', alpha=0.7)
        ax7.text(x_c[mitad], y_c[mitad], s_id, 
                    color='darkred', fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

        # Punto de origen (Centroide)
        centro = row['Geom_Grupo'].centroid
        ax7.scatter(centro.x, centro.y, color='darkblue', s=10)

# Leyenda manual
custom_lines = [
    Line2D([0], [0], color='orange', lw=4, alpha=0.3),
    Line2D([0], [0], color='blue', lw=1.2, linestyle='--'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=6)
]
ax7.legend(custom_lines, ['Grupo SAB T2', 'Corte Sectorial (Tráfico)', 'Centroide'], loc='upper right')

plt.title(f"Rediseño Tipo 2: Líneas de Corte Paralelas por Sector Adyacente")
plt.show()


def plot_diseno_sabs(df_sectores, df_original, df_final):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Dibujar los sectores de fondo (bordes grises)
    df_sectores.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5, label='Sectores')
    
    # 2. Dibujar el área de las celdas originales (SAB original) en azul claro
    df_original.plot(ax=ax, color='blue', alpha=0.2, edgecolor='blue', linestyle='--', label='SAB Original (Celdas)')
    
    # 3. Dibujar el diseño detallado final en naranja/rojo
    # Asegúrate de que df_final sea un GeoDataFrame
    
    gdf_final = gpd.GeoDataFrame(df_final, geometry='Geometry_Detallada')
    gdf_final.plot(ax=ax, color='red', alpha=0.6, edgecolor='black', linewidth=1.5, label='Diseño SAB Final')
    
    # Personalización del gráfico
    ax.set_title(f"Diseño de la forma de SAB del tipo 2\nFecha: {fecha_sel}-06-2022")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    
    # Crear una leyenda manual para mayor claridad
    
    custom_lines = [
        Line2D([0], [0], color='gray', lw=1),
        Line2D([0], [0], color='blue', lw=2, alpha=0.3, linestyle='--'),
        Line2D([0], [0], color='red', lw=2, alpha=0.7)
    ]
    ax.legend(custom_lines, ['Fronteras Sectores', 'SAB Original', 'SAB Final (Corte)'])
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Llamada a la función
plot_diseno_sabs(gdf_sectores, df_sabs2, df_final_geom)

gdf_sab2_f = gpd.GeoDataFrame(df_final_geom, geometry='Geometry_Detallada')


#%%
# ---------------------------------------------------------------------------------------- #
# -------------------------------- REPRESENTACIÓN GRÁFICA -------------------------------- #
# ---------------------------------------------------------------------------------------- #

print("\n--- GENERANDO REPRESENTACIÓN GRÁFICA ---")

## 1. Visualización final de la clasificación y filtrado de celdas SAB---------------------

# ANTES DEL FILTRO------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10)) 

# Dibujar todo el mallado de fondo
gdf_mallado.plot(ax=ax, color='whitesmoke', edgecolor='lightgrey', linewidth=0.2)

legend_handles = []

# --- 1. LÓGICA DE CONJUNTOS (Antes del filtro) ---
# Convertimos a sets para operar fácilmente con las coincidencias
# set_t1 = set(celdas_tipo1)
# set_t2 = set(celdas_tipo2)

# coincidentes = set_t1.intersection(set_t2)
# solo_t1 = set_t1 - coincidentes
# solo_t2 = set_t2 - coincidentes

# --- 2. GRAFICAR ---

# A) Tipo 1 Exclusivas (Azul)
mask_t1 = gdf_mallado['Cell_Name'].isin(tipo1_cells)
if mask_t1.any():
    gdf_mallado[mask_t1].plot(ax=ax, color='dodgerblue', alpha=0.7)
    legend_handles.append(mpatches.Patch(color='dodgerblue', alpha=0.7, label='Celdas de tipo 1'))

# B) Tipo 2 Exclusivas (Naranja)
mask_t2 = gdf_mallado['Cell_Name'].isin(tipo2_cells)
if mask_t2.any():
    gdf_mallado[mask_t2].plot(ax=ax, color='orange', alpha=0.7)
    legend_handles.append(mpatches.Patch(color='orange', alpha=0.7, label='Celdas de tipo 2'))

# C) Coincidentes (Rallado diagonal)
mask_coinc = gdf_mallado['Cell_Name'].isin(coincidencias)
if mask_coinc.any():
    # Usamos facecolor='cyan' (o el que prefieras) y edgecolor='black' para que el rallado se note bien
    gdf_mallado[mask_coinc].plot(ax=ax, facecolor='cyan', edgecolor='black', alpha=0.8, hatch='//')
    legend_handles.append(mpatches.Patch(facecolor='cyan', edgecolor='black', hatch='//', alpha=0.8, label='Celdas del tipo 1 y del tipo 2'))

# --- 3. CONFIGURACIÓN FINAL ---
ax.set_title(f"Distribución de Celdas antes de filtrar\nFecha: {fecha_sel}-06-2022")
ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')

# Colocar la leyenda fuera del mapa para que no estorbe
if legend_handles:
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

# DESPUÉS DEL FILTRO------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10)) 

# Dibujar todo el mallado de fondo
gdf_mallado.plot(ax=ax, color='whitesmoke', edgecolor='lightgrey', linewidth=0.2)

# Lista para gestionar manualmente los elementos de la leyenda
legend_handles = []

# 1. Tipo 1 y Coincidencias (Azul/Cian)
mask_t1 = gdf_mallado['Cell_Name'].isin(celdas_t1_vivas)
if mask_t1.any():
    gdf_mallado[mask_t1].plot(ax=ax, color='dodgerblue', alpha=0.8)
    legend_handles.append(mpatches.Patch(color='dodgerblue', label='Celdas de tipo 1'))

# 2. Tipo 2 que NO tocaban Tipo 1 (Naranja)
mask_t2_indep = gdf_mallado['Cell_Name'].isin(t2_no_tocan_t1)
if mask_t2_indep.any():
    gdf_mallado[mask_t2_indep].plot(ax=ax, color='orange', alpha=0.8)
    legend_handles.append(mpatches.Patch(color='orange', label='Celdas de tipo 2'))

# 3. Tipo 2 que SI tocaban T1 y son válidas (Verde)
mask_t2_val = gdf_mallado['Cell_Name'].isin(t2_validadas_contacto)
if mask_t2_val.any():
    gdf_mallado[mask_t2_val].plot(ax=ax, color='limegreen', alpha=0.8)
    legend_handles.append(mpatches.Patch(color='limegreen', label='Celdas de tipo 2 descartadas'))

# 4. Resaltar las descartadas (Rojo con patrón)
descartadas = t2_que_tocan_t1 - t2_validadas_contacto
mask_descartadas = gdf_mallado['Cell_Name'].isin(descartadas)
if mask_descartadas.any():
    gdf_mallado[mask_descartadas].plot(ax=ax, color='red', alpha=0.4, hatch='//')
    legend_handles.append(mpatches.Patch(facecolor='red', alpha=0.4, hatch='//', label='Celdas de tipo 2 descartadas'))

# Configuración final
plt.title(f"Clasificación y Filtrado de Celdas\nFecha: {fecha_sel}-06-2022 ")

# Crear la leyenda solo con los elementos que realmente se dibujaron
if legend_handles:
    plt.legend(handles=legend_handles, loc='upper right')

ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')
plt.show()
# ----------------------------------------------------------------------------------------

## 2. Fronteras rediseñadas para Tipo 1 (Simetría de Barrido)-----------------------------

# fig, ax5 = plt.subplots(figsize=(12, 10))

# #  Dibujar los Sectores del ACC (Polígonos de fondo)
# for index, row in DF_info_conf.iterrows():
#     poly = row['Contorno Sector']
#     x, y = poly.exterior.xy
#     ax5.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, label=f"Sector: {row['SECTOR_ID']}")
#     # Opcional: Añadir nombre del sector en el centro
#     ax5.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], fontsize=10, ha='center', weight='bold')

# #  Dibujar el fondo: Todo el mallado del ACC
# # Usamos un color neutro (gris claro) con bordes blancos para que sirva de contexto
# if not gdf_mallado.empty:
#     gdf_mallado.plot(ax=ax5, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6, label='Espacio Aéreo ACC')


# # Dibujar las nuevas fronteras de Tipo 1
# for flujo_orig, frontera_nueva in nuevas_fronteras_t1:
#     # El flujo original en rojo
#     ax5.plot(*flujo_orig.xy, color='red', linewidth=1.5, linestyle=':', label='Flujo T1')
#     # La nueva frontera barriendo hacia el lado opuesto
#     if frontera_nueva.geom_type == 'LineString':
#         ax5.plot(*frontera_nueva.xy, color='magenta', linewidth=3, label='Nueva Frontera Rediseñada')
#     elif frontera_nueva.geom_type == 'MultiLineString':
#         for part in frontera_nueva.geoms:
#             ax5.plot(*part.xy, color='magenta', linewidth=3)

# # Limpiar leyenda para no repetir etiquetas
# handles, labels = ax5.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax5.legend(by_label.values(), by_label.keys(), loc='upper right')

# plt.title(f"Rediseño de fronteras de los SAB del tipo 1 \nFecha: {fecha_sel}-06-2022")
# plt.show()

fig, ax5 = plt.subplots(figsize=(12, 10))

# 1. Dibujar los Sectores del ACC (SIN etiqueta 'label')
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    x, y = poly.exterior.xy
    # Quitamos el parámetro label=... para que no vaya a la leyenda
    ax5.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5)
    ax5.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6)

# 2. Dibujar el fondo: Todo el mallado del ACC
if not gdf_mallado.empty:
    gdf_mallado.plot(ax=ax5, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6)

# 3. Dibujar las nuevas fronteras de Tipo 1
for flujo_orig, frontera_nueva in nuevas_fronteras_t1:
    ax5.plot(*flujo_orig.xy, color='red', linewidth=1.5, linestyle=':')
    
    if frontera_nueva.geom_type == 'LineString':
        ax5.plot(*frontera_nueva.xy, color='magenta', linewidth=3)
    elif frontera_nueva.geom_type == 'MultiLineString':
        for part in frontera_nueva.geoms:
            ax5.plot(*part.xy, color='magenta', linewidth=3)

# 4. Crear LEYENDA MANUAL (Solo con los elementos de interés)
custom_lines_5 = [
    Line2D([0], [0], color='red', lw=1.5, linestyle=':', label='Flujos identificados'),
    Line2D([0], [0], color='magenta', lw=3, label='Línea de corte')
]
ax5.legend(handles=custom_lines_5, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax5.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel}-06-2022")
ax5.set_xlabel('Longitud [º]')
ax5.set_ylabel('Latitud [º]')
ax5.set_aspect('equal')

plt.tight_layout()
plt.show()

# # ----------------------------------------------------------------------------------------

# ## 3. Fronteras rediseñadas para Tipo 1 (Simetría de Barrido)-----------------------------

# fig, ax6 = plt.subplots(figsize=(12, 10))

# #  Dibujar los Sectores del ACC (Polígonos de fondo)
# for index, row in DF_info_conf.iterrows():
#     poly = row['Contorno Sector']
#     x, y = poly.exterior.xy
#     ax6.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, label=f"Sector: {row['SECTOR_ID']}")
#     # Opcional: Añadir nombre del sector en el centro
#     ax6.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], fontsize=10, ha='center', weight='bold')

# #  Dibujar el fondo: Todo el mallado del ACC
# # Usamos un color neutro (gris claro) con bordes blancos para que sirva de contexto
# if not gdf_mallado.empty:
#     gdf_mallado.plot(ax=ax6, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6, label='Espacio Aéreo ACC')

# if not df_sabs_t1_final.empty:
#     df_sabs_t1_final.plot(ax=ax6, color='cyan', alpha=0.5, edgecolor='blue', figsize=(10,10))

# # Limpiar leyenda para no repetir etiquetas
# handles, labels = ax6.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax6.legend(by_label.values(), by_label.keys(), loc='upper right')

# plt.title(f"Rediseño de fronteras de los SAB del tipo 1 \nFecha: {fecha_sel}-06-2022")
# plt.show()


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
ax6.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel}-06-2022")
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

# 3. Dibujar SABs Finales
if not df_sabs_t1_final.empty:
    # Eliminado el figsize interno que era redundante
    df_sabs_t1_final.plot(ax=ax8, color='cyan', alpha=0.5, edgecolor='blue')

gdf_sab2_f.plot(ax=ax8, color='red', edgecolor='black', linewidth=0.5, alpha=0.6)
    

# 4. Crear LEYENDA MANUAL
custom_lines_8 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=0.5, label='Forma de SAB del tipo 1'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Forma de SAB del tipo 2')
]

ax8.legend(handles=custom_lines_8, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax8.set_title(f"Diseño de las formas de SAB\nFecha: {fecha_sel}-06-2022")
ax8.set_xlabel('Longitud [º]')
ax8.set_ylabel('Latitud [º]')
ax8.set_aspect('equal')

plt.tight_layout()
plt.show()









#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- REPORTE Y GUARDADO DE DATOS -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# # Imprimir resultados del análisis
# print(f"Total celdas Tipo 1 iniciales: {len(tipo1_cells)}")
# print(f"Total SABs Agrupados (Tipo 2): {len(tipo2_cells)}")
# print(f"Coincidencias encontradas (Definitivas T1): {len(coincidencias)}")

# print("\n--- RESULTADO FINAL ---")
# # Aquí mostramos cuántos polígonos finales reales han quedado tras la agrupación
# num_sabs_finales = len(gdf_sab_definitivo_filtrado) if not gdf_sab_definitivo_filtrado.empty else 0
# print(f"Número total de SABs físicos tras agrupar y filtrar cuñas: {num_sabs_finales}")

# # Guardar resultados a CSV (como lo tenías)
# resultados = {
#     'SAB_ID': list(tipo2_cells) + evaluar_tipo1,
#     'Tipo': ['Tipo 2 (Definitivo)'] * len(tipo2_cells) + ['Tipo 1 (A evaluar)'] * len(evaluar_tipo1),
#     'Estado': ['Definitivo'] * len(tipo2_cells) + ['Pendiente Evaluación'] * len(evaluar_tipo1)
# }
# df_final = pd.DataFrame(resultados)
# df_final.to_csv(os.path.join(PATH_sabs, f'sabs_clasificados_finales_{fecha_sel}.csv'), index=False)
# print("Archivo CSV de clasificación guardado.")

