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
PATH_SECTOR_DATA = 'F:\\Users\\Lai\\Datos\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'
PATH_flujos = 'F:\\Users\\Lai\\original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\'
PATH_mallado = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

# # Enero
PATH_TRAFICO_CELDA = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Enero2022\\'
PATH_resultados = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\Enero2022\\test\\'
PATH_TRAFICO = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\test\\Enero2022\\'
PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\Enero2022\\'
PATH_COMPLEJIDAD = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Enero2022\\' # Asegúrate de que esta ruta es correcta

# Junio
# PATH_TRAFICO_CELDA = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Datos de entrada eCOMMET\\mallado_20x20\\'
# PATH_resultados = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\Junio2022\\test\\'
# PATH_TRAFICO = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\Junio2022\\'
# PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\Junio2022\\'
# PATH_COMPLEJIDAD = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Junio2022\\test\\' # Asegúrate de que esta ruta es correcta


# configuración del estudio (ejemplo: CNF5A, CNF5B, etc.)
#! PENDIENTE DE AUTOMATIZAR LA SELECCIÓN DE CONFIGURACIÓN (ej. con input o argumentos)
configuracion_estudio = 'CNF5A'

# Día seleccionado

mes_sel = 1
dia_sel = 4
#mes_sel = input("Introduce un mes de 2022 (MM): ")
#dia_sel = input(f"Introduce un día del {mes_sel} de 2022 (2022-{mes_sel}-DD): ")

fecha_sel = f"{mes_sel:02d}-{dia_sel:02d}"  # Formatear con ceros a la izquierda (ej. 01, 02, ..., 30) 

# Crear el objeto de fecha y el nombre de la carpeta (formato YYYY-MM-DD)
fecha_data = datetime(2022, mes_sel, dia_sel)  # Extraemos el día del string formateado
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
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda_{dia_sel:02d}_filtrado.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo_{dia_sel:02d}_filtrado.pkl')

# Datos de los sabs detectados
sab_tipo1 = pd.read_pickle(PATH_sabs + f'celdas_sab_tipo1_2022-{fecha_sel}.pkl')
sab_tipo2 = pd.read_pickle(PATH_sabs + f'SAB_tipo2_2022-{fecha_sel}.pkl')

# Datos del mallado de tamaño 20x20 nm
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')

# Datos de los sectores (con su geometría)
DF_info_conf = pd.read_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')

# Datos de los flujos clusterizados
df_espinazos = pd.read_pickle(PATH_sabs + f'espinazos_flujos_{dia_sel:02d}.pkl')

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
df_complejidad_sum = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Suma_2022-{fecha_sel}_Completo.pkl')
df_complejidad_media = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Media_2022-{fecha_sel}_Diaria.pkl')

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

# 2. Evaluación !! Tipo 1 !! por Complejidad
evaluar_tipo1 = list(tipo1_cells)
# UMBRAL definido previamente (ej: 30.0)
tipo1_validadas = filtrar_sabs_por_complejidad(evaluar_tipo1, df_complejidad_sum, UMBRAL)

# Identificar flujos a eliminar (por pasar por celdas T1 complejas)
tipo1_descartadas = set(evaluar_tipo1) - set(tipo1_validadas)
flujos_a_eliminar = sab_tipo1[sab_tipo1['Cell_Name'].isin(tipo1_descartadas)]['Flujo'].unique()

# Filtrar SAB Tipo 1 final (usamos .copy() para evitar SettingWithCopyWarning)
sab1_final = sab_tipo1[~sab_tipo1['Flujo'].isin(flujos_a_eliminar)].copy()
flujo_localizados = sab1_final['Flujo'].unique()
celdas_t1_vivas = set(sab1_final['Cell_Name'])

sab1_final.to_pickle(PATH_sabs + f'SAB_tipo1_final_2022-{fecha_sel}.pkl')
sab1_final.to_csv(PATH_sabs + f'SAB_tipo1_final_2022-{fecha_sel}.csv', index=False)

# Unión de T1 validadas para detectar contacto
union_t1 = gdf_mallado[gdf_mallado['Cell_Name'].isin(celdas_t1_vivas)].union_all()

sabs_finales_sistema = list(celdas_t1_vivas)


# 6. RECALCULAR sabs_finales_sistema 
# Los SABs finales totales serían:
# (Los de Tipo 2) + (Los T1 evaluados que no eran complejos) + (Los T2 que conectan con los T1 pero tiene suficiente tamaño)
sabs_finales_sistema = list(celdas_t1_vivas) 

print('3')
## analizar los flujos para ver si son coincidentes
# 
dist_max = 0.5
ang_max = 10.0

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

import math

def calcular_angulo_entre_lineas(linea1, linea2):
    # Extraer coordenadas de inicio y fin de la primera línea
    coords1 = list(linea1.coords)
    p1_inicio, p1_fin = coords1[0], coords1[-1]
    
    # Extraer coordenadas de inicio y fin de la segunda línea
    coords2 = list(linea2.coords)
    p2_inicio, p2_fin = coords2[0], coords2[-1]
    
    # Calcular el ángulo de cada vector respecto al eje X usando atan2(y, x)
    angulo1 = math.atan2(p1_fin[1] - p1_inicio[1], p1_fin[0] - p1_inicio[0])
    angulo2 = math.atan2(p2_fin[1] - p2_inicio[1], p2_fin[0] - p2_inicio[0])
    
    # Calcular la diferencia absoluta en grados
    dif_angulo = math.degrees(abs(angulo1 - angulo2))
    
    # Normalizar para que el ángulo siempre esté entre 0 y 180 grados
    if dif_angulo > 180:
        dif_angulo = 360 - dif_angulo
        
    return dif_angulo

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
        print(f'Comparando {f_tag} con {f_c}')
        print(f'comparar. {f_comp_line}')

        # Cálculo de métricas
        dist = f_line.distance(f_comp_line)
        angulo = calcular_angulo_entre_lineas(f_line, f_comp_line)
        
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
                    'Tipo': '1'
                })

print('3')
# --- CREACIÓN DEL DATAFRAME FINAL ---
df_sabs_t1_final = gpd.GeoDataFrame(sabs_tipo1_definitivos, geometry='geometry', crs=gdf_mallado.crs)
df_sabs_t1_final.to_pickle(PATH_sabs + f'SAB_tipo1_definitivo_2022-{fecha_sel}.pkl')
df_sabs_t1_final.to_csv(PATH_sabs + f'SAB_tipo1_definitivo_2022-{fecha_sel}.csv', index=False)

print(f"Se han generado {len(df_sabs_t1_final)} SABs definitivos de Tipo 1.")

# Visualización rápida para verificar
if not df_sabs_t1_final.empty:
    ax = df_sabs_t1_final.plot(color='cyan', alpha=0.5, edgecolor='blue', figsize=(10,10))
    plt.title("Geometrías Finales SABs Tipo 1")
    plt.show()

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


# --- 3. CONFIGURACIÓN FINAL ---
ax.set_title(f"Distribución de Celdas antes de filtrar\nFecha: {fecha_sel}-2022")
ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')

# Colocar la leyenda fuera del mapa para que no estorbe
if legend_handles:
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

plt.tight_layout()
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

# plt.title(f"Rediseño de fronteras de los SAB del tipo 1 \nFecha: {fecha_sel}-2022")
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
ax5.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel}-2022")
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

# plt.title(f"Rediseño de fronteras de los SAB del tipo 1 \nFecha: {fecha_sel}-2022")
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
ax6.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel}-2022")
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

df_sabs_t2.plot(ax=ax8, color='red', edgecolor='black', linewidth=0.5, alpha=0.6)
    

# 4. Crear LEYENDA MANUAL
custom_lines_8 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=0.5, label='Forma de SAB del tipo 1'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Forma de SAB del tipo 2')
]

ax8.legend(handles=custom_lines_8, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax8.set_title(f"Diseño de las formas de SAB\nFecha: {fecha_sel}-2022")
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

