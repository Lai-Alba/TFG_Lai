#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import itertools
import warnings
import re
import os
import pickle
import shap
import gc
import ast
import math
import time as t

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import seaborn as sns

import time
from datetime import datetime

import itertools
from itertools import product

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import geopy.distance
from geopy.distance import geodesic

import shapely
from shapely.geometry import box
from shapely.geometry import Point, LineString, box, MultiLineString
from shapely.geometry import Polygon, MultiPolygon 
from shapely.ops import nearest_points
from shapely.ops import unary_union
from shapely.wkt import loads

from sklearn.cluster import DBSCAN
from collections import Counter
from IPython.display import display
from datetime import datetime, time


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

### Parámetros de estudio

# configuración de estudio
configuracion_estudio = 'CNF5A'

# Día seleccionado
dia_sel = input(f"Introduce un día del {mes_input} de 2022 (2022-{mes_sel}-DD): ")
dia_sel = f"{int(dia_sel):02d}"  # Asegura que el día tenga dos dígitos

fecha_sel = f"2022-{mes_sel}-{dia_sel}"

### franja de estudio
hora_ini = time(0, 0, 0) 
hora_fin = time(23, 59, 59) 
hora_ini_str = '00'
hora_fin_str = '24'    

# Para los resultados de segundo tipo no hace analizar por horas
# pero por si acaso quiere analizar por horas, se deja la opción de elegir horas

dia_horas = 'dia' # por defecto, el análisis es por día completo
# dia_horas = input("Introduce si quiere estudiar un día entero o una franja de horas (dia/hora): ").strip().lower()

# if dia_horas == 'dia':
#     hora_ini = time(0, 0, 0) 
#     hora_fin = time(23, 59, 59) 
#     hora_ini_str = '00'
#     hora_fin_str = '24'    
# elif dia_horas == 'hora':
#     hora_ini_str = input("Introduce la hora de inicio (ej. 08, 12): ")
#     hora_ini_str = f"{int(hora_ini_str):02d}"  # Asegura que la hora tenga dos dígitos
#     hora_fin_str = input("Introduce la hora de fin (ej. 08, 12): ")
#     hora_fin_str = f"{int(hora_fin_str):02d}"
#     hora_ini = time(int(hora_ini_str), 0, 0) 
#     hora_fin = time(int(hora_fin_str), 0, 0) 
    
# horas para el analisis manual---------------------
print(f"la fecha de estudio es {fecha_sel}")

# Extraemos el día del string formateado
nombre_carpeta = f"RESULTADOS_{fecha_sel}"  # Nombre de la carpeta para guardar resultados

# 4. Unir la ruta base con la nueva carpeta
PATH_COMPLEJIDAD_DIA = os.path.join(PATH_COMPLEJIDAD, nombre_carpeta)

# Tamaño de celda en nm
cell_size_nm = 20
# cell_size_nm = input("Introduce el tamaño de celda en NM (ej. 20): ")

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

## -------------------------------------------------------------------------------------------------------------------- #

# 1. Cargar Datos y Parsear Geometrías

# DATASET ANÁLISIS FLUJOS POR CELDA: qué flujos atraviesan cada celda
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo.pkl')

# DATASET DE TRÁFICO REAL: qué vuelos reales hay en el día seleccionado
DF_Trafico = pd.read_pickle(PATH_TRAFICO + f'dataset_vuelos_reales_{fecha_sel}.pkl')

# Dataset de la geometría de los sectores
DF_info_conf = pd.read_pickle(PATH_sabs + f'{configuracion_estudio}_DF_info_conf.pkl')
# y su mallado
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')
mallado_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_DF_cells.pkl')

# DATASET DE CELDAS FRONTERIZAS: qué celdas están en la frontera entre sectores
df_border_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_border_cells_{cell_size_nm}.pkl')

# Geometría
DF_Flujos['geometry'] = DF_Flujos['Line']

gdf_mallado = gpd.GeoDataFrame(df_mallado, geometry='Polygon')
gdf_mallado = gdf_mallado.rename_geometry('geometry')

gdf_sectores = gpd.GeoDataFrame(
    DF_info_conf.rename(columns={'Contorno Sector': 'geometry'}),
    geometry='geometry',
    crs=gdf_mallado.crs
)

# Obtención de las fronteras internas
contorno_externo_global = gdf_sectores.union_all().boundary
fronteras_internas = []

for _, sector in gdf_sectores.iterrows():
    sector_id = sector['SECTOR_ID']
    geometria_sector = sector['geometry']

    perimetro_sector = geometria_sector.boundary

    solo_interno = perimetro_sector.difference(contorno_externo_global.buffer(1e-7))
    
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

print(f"Fronteras internas encontradas: {fronteras_internas}")
gdf_fronteras_internas = gpd.GeoDataFrame(fronteras_internas, geometry='geometry', crs=gdf_sectores.crs)
print(f"Procesados {len(gdf_fronteras_internas)} sectores con fronteras internas.")

# filtado de datos
#Función de filtrado
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


##

## Datos de complejidad----------------------------------------------------------------------------

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


#%%

# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- AGRUPAR FLUJOS POR SECTORES ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# Funciones
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
                flujo_id = flujo_row['Flujo_Clusterizado']

                # Asegurarnos de que el resultado sea una LineString (a veces devuelve MultiLineString)
                if fragmento.geom_type == 'LineString':
                    flujos_en_sector.append({'Line': fragmento, 'Original_ID': flujo_id, 'Sector_ID': sector_id})
                elif fragmento.geom_type == 'MultiLineString':
                    for part in fragmento.geoms:
                        flujos_en_sector.append({'Line': part, 'Original_ID': flujo_id, 'Sector_ID': sector_id})

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
            linea_mas_larga = max(subset_lines, key=lambda linea: linea.length)
            longitud_max = linea_mas_larga.length

            backbones_local[c_id] = generar_segmento_representativo(subset_lines, longitud_segmento=longitud_max)
        
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

    return df_final, df_espinazos

# EJECUCIÓN
df_resultado_sectores, df_espinazos_final = agrupar_por_sectores(df_flujos, DF_info_conf, dist_max_km=0.5, ang_max=5, min_flujos=5)
df_espinazos_final.to_csv(PATH_sabs + f'espinazos_flujos_{fecha_sel}_{h_ini_str}_{h_fin_str}_test.csv', index=False)
df_espinazos_final.to_pickle(PATH_sabs + f'espinazos_flujos_{fecha_sel}_{h_ini_str}_{h_fin_str}_test.pkl')
print("Guardado")

def plot_mapa_flujos_global(gdf_sectores, df_flujos_agrupados, df_espinazos):
    """
    Representa el mapa global de los sectores del ACC junto con los flujos 
    originales agrupados y sus líneas representativas (espinazos).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
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
    ax.set_title(f"Mapa de flujos agrupados por sectores\nFecha: {fecha_sel}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

plot_mapa_flujos_global(gdf_sectores, df_resultado_sectores, df_espinazos_final)

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- Análisis de complejidad ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #





#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- ENCONTRAR PUNTOS DE CRUCE ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# 1. Encontrar Puntos de Cruce Geométrico Exacto
intersections = []

# Pts de cruce de los flujos clsterizados----------------------------------------------
lines = df_espinazos_final['Representative_Line'].dropna().tolist()
flow_ids = df_espinazos_final['cluster_global'].dropna().tolist()

line_intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        if lines[i].intersects(lines[j]):
            inter = lines[i].intersection(lines[j])
            if inter.geom_type == 'Point':
                intersections.append((inter.x, inter.y))
                line_intersections.append((flow_ids[i], flow_ids[j], inter.x, inter.y))
            elif inter.geom_type == 'MultiPoint':
                for p in inter.geoms:
                    intersections.append((p.x, p.y))
                    line_intersections.append((flow_ids[i], flow_ids[j], p.x, p.y))


# Pts de cruce de todos los flujos ----------------------------------------------
# lines = df_flujos['geometry'].dropna().tolist()
# flow_ids = df_flujos['Clave_Flujo'].dropna().tolist()

# line_intersections = []
# for i in range(len(lines)):
#     for j in range(i + 1, len(lines)):
#         if lines[i].intersects(lines[j]):
#             inter = lines[i].intersection(lines[j])
#             if inter.geom_type == 'Point':
#                 intersections.append((inter.x, inter.y))
#                 line_intersections.append((flow_ids[i], flow_ids[j], inter.x, inter.y))
#             elif inter.geom_type == 'MultiPoint':
#                 for p in inter.geoms:
#                     intersections.append((p.x, p.y))
#                     line_intersections.append((flow_ids[i], flow_ids[j], p.x, p.y))


# 2. Agrupar puntos (Clustering) y calcular densidad y polígonos
pts = np.array(intersections)
clustering = DBSCAN(eps=0.075, min_samples=1).fit(pts) 

labels = clustering.labels_
cluster_centers = {
    lbl: pts[labels == lbl].mean(axis=0) 
    for lbl in set(labels)
}

cluster_counts = Counter(labels)

# --- Calcular el polígono (envolvente convexa) de cada cluster ---
cluster_polygons = {}
from shapely.geometry import MultiPoint

for label, centroid in cluster_centers.items():
    cluster_points = pts[labels == label]
    
    # Si hay 1 o 2 puntos, un Convex Hull es un punto o una línea. 
    # Usamos buffer() para inflarlo y convertirlo en un polígono visible.
    if len(cluster_points) < 3:
        poly = MultiPoint(cluster_points).buffer(0.015)
    else:
        # Envolvente convexa para adaptar la forma, más un buffer mínimo para estética
        poly = MultiPoint(cluster_points).convex_hull.buffer(0.005)
        
    cluster_polygons[label] = poly

# flow_to_clusters = {fid: [] for fid in flow_ids}
# for (fid1, fid2, x, y), label in zip(line_intersections, labels):
#     flow_to_clusters[fid1].append((label, x, y))
#     flow_to_clusters[fid2].append((label, x, y))

# --- (Tu código anterior hasta la asignación de flow_to_clusters) ---

flow_to_clusters = {fid: [] for fid in flow_ids}
# Añadimos un diccionario inverso para saber qué flujos cruzan cada clúster
cluster_to_flows = {lbl: set() for lbl in set(labels)}

for (fid1, fid2, x, y), label in zip(line_intersections, labels):
    flow_to_clusters[fid1].append((label, x, y))
    flow_to_clusters[fid2].append((label, x, y))
    # Asignamos los flujos al clúster correspondiente
    cluster_to_flows[label].add(fid1)
    cluster_to_flows[label].add(fid2)

# Crear grafo con NetworkX asignando pesos, el polígono y los flow_ids a los nodos
G = nx.Graph()
for label, centroid in cluster_centers.items():
    weight = cluster_counts[label]
    poly = cluster_polygons[label]
    # Convertimos el set a lista para guardarlo en el nodo
    flujos_involucrados = list(cluster_to_flows[label]) 
    
    # Añadimos el atributo 'flow_ids' al nodo
    G.add_node(label, pos=(centroid[0], centroid[1]), weight=weight, polygon=poly, flow_ids=flujos_involucrados)

# --- (El resto de tu lógica para añadir los edges a G se mantiene igual) ---

# Crear grafo con NetworkX asignando pesos y el polígono a los nodos
# G = nx.Graph
# for label, centroid in cluster_centers.items():
#     weight = cluster_counts[label]
#     poly = cluster_polygons[label]
#     G.add_node(label, pos=(centroid[0], centroid[1]), weight=weight, polygon=poly)

for fid, points in flow_to_clusters.items():
    if len(points) > 1:
        line_idx = flow_ids.index(fid)
        start_pt = Point(lines[line_idx].coords[0])
        sorted_points = sorted(points, key=lambda p: start_pt.distance(Point(p[1], p[2])))
        
        for i in range(len(sorted_points) - 1):
            n1, n2 = sorted_points[i][0], sorted_points[i+1][0]
            if n1 != n2:
                G.add_edge(n1, n2)

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- DETECCIÓN DE GRUPOS DE PTS DE CRUCE ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# 1. Definir parámetros de filtrado
# Recuerda ajustar 'umbral_distancia' según el sistema de coordenadas (CRS). 
# Si usas grados, ~0.05. Si usas metros, prueba con valores como 5000 (5km).
umbral_distancia = 0.25
peso_minimo = 2

# Estructuras para almacenar los resultados
nodos_totales_a_eliminar = set() # Usamos set para evitar duplicados si un clúster toca dos sectores
nodos_cercanos = []
# 2. Iterar por cada sector único en el GeoDataFrame de fronteras internas
sectores_unicos = gdf_fronteras_internas['SECTOR_ID'].unique()

print(f"\nIniciando análisis de proximidad para {len(sectores_unicos)} sectores...")

for sector_id in sectores_unicos:
    # Agrupar todas las geometrías de frontera interna de este sector en una sola
    lineas_sector = gdf_fronteras_internas[gdf_fronteras_internas['SECTOR_ID'] == sector_id]['geometry']
    frontera_interna_unida = lineas_sector.union_all()
    
    # 3. Evaluar cada nodo (clúster) del grafo
    for nodo, datos in G.nodes(data=True):
        peso_cluster = datos['weight']
        
        # Evaluar primero el peso para ahorrar cálculos geométricos pesados
        if peso_cluster > peso_minimo:
            poligono_cluster = datos['polygon']
            flujos_del_nodo = datos['flow_ids']

            # Calcular la distancia mínima entre el polígono del clúster y la frontera interna
            distancia = poligono_cluster.distance(frontera_interna_unida)
            
            # Si cumple la condición de cercanía, lo marcamos
            if distancia <= umbral_distancia:
                nodos_cercanos.append({
                        'nodo': nodo,
                        'peso': peso_cluster,
                        'distancia': distancia,
                        'geometry': poligono_cluster,
                        'flow_ids': flujos_del_nodo, # <-- Los guardamos aquí
                        'sector_id': sector_id
                    })
                
                
    # Guardar un registro por sector (útil para debuggear o pintar en mapas luego)
    if nodos_cercanos:
        print(f"  -> Sector {sector_id}: Detectados {len(nodos_cercanos)} clusters cercanos.")

print(f"Los flujos identificados son: {nodos_cercanos[0]['flow_ids']}y {nodos_cercanos[1]['flow_ids']}")

print("\n--- RESUMEN DE LIMPIEZA ---")



# 1. Obtener la lista única de todos los flujos involucrados en los nodos cercanos
flujos_cercanos_ids = set()
poligonos_cercanos = []

for nc in nodos_cercanos:
    flujos_cercanos_ids.update(nc['flow_ids'])
    poligonos_cercanos.append(nc['geometry'])
    
# 2. Filtrar el dataframe original para obtener solo las líneas de esos flujos
df_flujos_cercanos = df_espinazos_final[df_espinazos_final['cluster_global'].isin(flujos_cercanos_ids)]

# Crear un GeoDataFrame temporal para los polígonos de los clústers (facilita el ploteo)
gdf_clusters_cercanos = gpd.GeoDataFrame(geometry=poligonos_cercanos, crs=gdf_fronteras_internas.crs)

# 3. Configurar el Plot
fig, ax = plt.subplots(figsize=(12, 10))
gdf_mallado.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.15)

for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    if poly.geom_type == 'Polygon':
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
    elif poly.geom_type == 'MultiPolygon':
        for p in poly.geoms:
            x, y = p.exterior.xy
            ax.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
            
    ax.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6, zorder=3)

# a) Pintar fronteras internas (contexto base)
gdf_fronteras_internas.plot(ax=ax, color='none', edgecolor='black', linewidth=1, linestyle='--', label='Fronteras Internas')

# b) Pintar los flujos cercanos filtrados
# Si df_espinazos_final no es un GeoDataFrame que use 'Representative_Line' como columna de geometría activa, 
# asegúrate de establecerla antes: 
df_flujos_cercanos = df_flujos_cercanos.set_geometry('Representative_Line')

df_flujos_cercanos.plot(ax=ax, color='blue', linewidth=2, alpha=0.7, label='Flujos Cercanos')

# c) Pintar los polígonos de los clústers identificados
if not gdf_clusters_cercanos.empty:
    gdf_clusters_cercanos.plot(ax=ax, color='red', alpha=0.5, edgecolor='darkred', label='Clústers Cercanos')

# Ajustes estéticos
plt.title(f'Clústers identificados \n Fecha: {fecha_sel}', fontsize=14)
plt.xlabel('Longitud')
plt.ylabel('Latitud')

# Crear leyenda manual si es necesario (ya que plotear sobre el mismo ax a veces pisa las labels)
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# frontera_patch = mlines.Line2D([], [], color='black', linestyle='--', label='Fronteras')
flujo_patch = mlines.Line2D([], [], color='blue', linewidth=2, label='Grupos de flujos identificados')
# cluster_patch = mpatches.Patch(color='red', alpha=0.5, label='Polígonos Clúster')

plt.legend(handles=[flujo_patch], loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print("-" * 27)

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------- ANÁLISIS DETALLADO DE FLUJOS ORIGINALES EN NODOS DE CRUCE -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# 1. NOMBRAR/ENUMERAR LOS NODOS DETECTADOS
# Extraemos los IDs únicos de los nodos que pasaron tu filtro de proximidad (o usamos todos si prefieres)
nodos_validos_ids = set([d['nodo'] for d in nodos_cercanos])

# Creamos un diccionario para mapear el ID de DBSCAN a un ID numérico secuencial (1, 2, 3...)
mapeo_nodos = {nodo_original: i + 1 for i, nodo_original in enumerate(nodos_validos_ids)}

print(f"\n--- ANALIZANDO LOS {len(mapeo_nodos)} NODOS DE CRUCE PRINCIPALES ---")

# 2. IDENTIFICAR QUÉ GRUPOS DE FLUJOS (cluster_global) PARTICIPAN EN CADA NODO
# Creamos un diccionario: {nodo_id: set(cluster_global_1, cluster_global_2, ...)}
nodos_a_grupos = {nodo: set() for nodo in nodos_validos_ids}

# Recorremos las intersecciones de las líneas representativas para ver qué grupos forman cada nodo
for (fid1, fid2, x, y), label in zip(line_intersections, labels):
    if label in nodos_validos_ids:
        nodos_a_grupos[label].add(fid1)
        nodos_a_grupos[label].add(fid2)

# 3. EXTRAER FLUJOS, CALCULAR CRUCES REALES Y REPRESENTAR
# Parámetros para el DBSCAN de los puntos de cruce reales
# Ajusta eps_cruces según tu CRS (0.01 a 0.05 suele funcionar bien para grados EPSG:4326)
eps_cruces = 0.1
min_samples_cruces = 20

# ---> NUEVO: CREAR LA FIGURA GLOBAL ANTES DEL BUCLE <---
fig_global, ax_global = plt.subplots(figsize=(10, 8))
global_bounds = [] # Para calcular el zoom global al final

# gdf_mallado.plot(ax=ax_global, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6, zorder=1)


# for index, row in DF_info_conf.iterrows():
#     poly = row['Contorno Sector']
#     if poly.geom_type == 'Polygon':
#         x, y = poly.exterior.xy
#         ax_global.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
#     elif poly.geom_type == 'MultiPolygon':
#         for p in poly.geoms:
#             x, y = p.exterior.xy
#             ax_global.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
            
#     ax_global.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
#              fontsize=10, ha='center', weight='bold', alpha=0.6, zorder=3)

sectores_dibujados_global = set()

for nodo_original, nodo_nuevo_id in mapeo_nodos.items():
    grupos_implicados = nodos_a_grupos[nodo_original]
    
    # --- A. Extracción de Flujos Originales ---
    flujos_originales_nodo = df_resultado_sectores[
        df_resultado_sectores['cluster_global'].isin(grupos_implicados)
    ]['Line'].tolist()
    
    # --- B. Calcular Cruces Reales ---
    puntos_cruce_reales = []
    
    for i in range(len(flujos_originales_nodo)):
        for j in range(i + 1, len(flujos_originales_nodo)):
            if flujos_originales_nodo[i].intersects(flujos_originales_nodo[j]):
                inter = flujos_originales_nodo[i].intersection(flujos_originales_nodo[j])
                if inter.geom_type == 'Point':
                    puntos_cruce_reales.append(inter)
                elif inter.geom_type == 'MultiPoint':
                    for p in inter.geoms:
                        puntos_cruce_reales.append(p)

    sector_id_nodo = None
    frontera_interna_unida = None
    
    if len(puntos_cruce_reales) > 0:
        
        # 1. Calculamos el "centro de gravedad" de todos los cruces de este nodo
        centro_nodo = MultiPoint(puntos_cruce_reales).centroid
        
        # 2. Comprobamos físicamente en qué sector cae ese punto exacto
        # Nota: Asegúrate de usar tu GeoDataFrame de sectores (gdf_sectores o DF_info_conf)
        for _, row_sector in gdf_sectores.iterrows():
            geometry_DGI = DF_info_conf[DF_info_conf['SECTOR_ID'] == 'LECMDGI']['Contorno Sector'].values[0]
            geometry_BLI = DF_info_conf[DF_info_conf['SECTOR_ID'] == 'LECMBLI']['Contorno Sector'].values[0]
            # Revisamos si el polígono del sector contiene el punto de cruce
            # Si en tu dataframe la columna geométrica se llama de otra forma (ej. 'Contorno Sector'), cámbiala aquí
            if row_sector.geometry.contains(centro_nodo):
                sector_id_nodo = row_sector['SECTOR_ID']
                if sector_id_nodo == 'LECMBLI' and geometry_DGI.contains(centro_nodo):
                    sector_id_nodo = 'LECMDGI'
                    # por el área de solape entre estos sectores. Se prioriza DGI si el centro cae dentro de su contorno.
                    # porque el punto caliente en este caso pertenece al sector DGI
                break
                
        
        # 3. Si encontramos el sector, extraemos su frontera interna, lo dibujamos y buscamos vecinos
        if sector_id_nodo:
            # Extraer el polígono principal
            poly_principal = DF_info_conf[DF_info_conf['SECTOR_ID'] == sector_id_nodo]['Contorno Sector'].values[0]
            
            # --- DIBUJAR SECTOR PRINCIPAL (Si no se ha dibujado antes) ---
            if sector_id_nodo not in sectores_dibujados_global:
                if poly_principal.geom_type == 'Polygon':
                    x, y = poly_principal.exterior.xy
                    ax_global.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
                elif poly_principal.geom_type == 'MultiPolygon':
                    for p in poly_principal.geoms:
                        x, y = p.exterior.xy
                        ax_global.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
                        
                ax_global.text(poly_principal.centroid.x, poly_principal.centroid.y, sector_id_nodo, 
                        fontsize=10, ha='center', weight='bold', alpha=0.8, zorder=3) 
                
                sectores_dibujados_global.add(sector_id_nodo)
            
            # --- BUSCAR Y DIBUJAR SECTORES VECINOS ---
            # Iteramos sobre todos los sectores disponibles para ver cuáles tocan al principal
            for _, row_posible_vecino in DF_info_conf.iterrows():
                id_vecino = row_posible_vecino['SECTOR_ID']
                poly_vecino = row_posible_vecino['Contorno Sector']
                
                # Es vecino si intersecta/toca al principal, y no es el principal en sí
                if id_vecino != sector_id_nodo and id_vecino not in sectores_dibujados_global:
                    if poly_principal.intersects(poly_vecino):
                        # Lo dibujamos con un estilo diferente (más claro, borde punteado gris)
                        if poly_vecino.geom_type == 'Polygon':
                            x, y = poly_vecino.exterior.xy
                            ax_global.fill(x, y, alpha=0.15, edgecolor='black', linestyle='--', linewidth=1.0, zorder=1)
                        elif poly_vecino.geom_type == 'MultiPolygon':
                            for p in poly_vecino.geoms:
                                x, y = p.exterior.xy
                                ax_global.fill(x, y, alpha=0.15, edgecolor='black', linestyle='--', linewidth=1.0, zorder=1)
                                
                        # Etiqueta del vecino un poco más discreta
                        ax_global.text(poly_vecino.centroid.x, poly_vecino.centroid.y, id_vecino, 
                                fontsize=8, ha='center', style='italic', alpha=0.8, zorder=3) 
                        
                        sectores_dibujados_global.add(id_vecino) # Añadimos al registro para no redibujarlo

            lineas_sector = gdf_fronteras_internas[gdf_fronteras_internas['SECTOR_ID'] == sector_id_nodo]['geometry']
            frontera_interna_unida = lineas_sector.union_all()
                        
    # --- C. Clustering (DBSCAN) de los Puntos de Cruce ---
    labels_cruces = []
    n_clusters_cruces = 0
    
    if len(puntos_cruce_reales) > 0:
        # Extraer coordenadas [x, y] para DBSCAN
        coords_cruces = np.array([[p.x, p.y] for p in puntos_cruce_reales])
        
        # Aplicar DBSCAN
        db_cruces = DBSCAN(eps=eps_cruces, min_samples=min_samples_cruces).fit(coords_cruces)
        labels_cruces = db_cruces.labels_
        
        # Ignorar el ruido (-1) para contar los clusters válidos
        n_clusters_cruces = len(set(labels_cruces)) - (1 if -1 in labels_cruces else 0)
                        
    # --- D. Representación Gráfica (Zoom al Nodo) ---
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Dibujar los flujos originales
    for linea in flujos_originales_nodo:
        ax.plot(*linea.xy, color='steelblue', alpha=0.3, linewidth=1.5)
        ax_global.plot(*linea.xy, color='steelblue', alpha=0.15, linewidth=1.0)
        global_bounds.append(linea.bounds)

    # 2. Dibujar el polígono original del nodo 
    poly_nodo = G.nodes[nodo_original]['polygon']
    if poly_nodo.geom_type == 'Polygon':
        x_poly, y_poly = poly_nodo.exterior.xy
        ax.plot(x_poly, y_poly, color='darkorange', linewidth=2, linestyle='--', label='Área Original del Nodo')
        ax_global.plot(x_poly, y_poly, color='darkorange', linewidth=2, linestyle='--', label='Área Original de Nodos' if nodo_nuevo_id == 1 else "")

    # 3. Dibujar los puntos de cruce reales y contar por grupo
    total_cruces = len(puntos_cruce_reales)
    
    if total_cruces > 0:
        cmap = cm.get_cmap('tab20', max(1, n_clusters_cruces))
        
        # Dibujar los puntos primero
        for idx, p in enumerate(puntos_cruce_reales):
            label = labels_cruces[idx]
            if label == -1: # Ruido
                ax.scatter(p.x, p.y, color='gray', s=10, alpha=0.5, marker='x')
                ax_global.scatter(p.x, p.y, color='gray', s=5, alpha=0.3, marker='x')
            else: # Pertenecen a un sub-cluster
                ax.scatter(p.x, p.y, color=cmap(label), s=30, alpha=0.7, zorder=5)
                ax_global.scatter(p.x, p.y, color=cmap(label), s=20, alpha=0.6, zorder=5)

        # 4. Calcular el número de cruces por sub-grupo y mostrarlo en el mapa
        etiquetas_unicas = set(labels_cruces)
        for label in etiquetas_unicas:
            # Filtrar los puntos que pertenecen a este label
            puntos_grupo = [puntos_cruce_reales[i] for i, l in enumerate(labels_cruces) if l == label]
            cantidad = len(puntos_grupo)
            
            if label == -1:
                # Opcional: Agregar info de puntos aislados a la leyenda
                ax.scatter([], [], color='gray', marker='x', label=f'Cruces aislados (Ruido): {cantidad}')
            else:
                # Calcular el centro (promedio de x e y) para poner la etiqueta
                cx = np.mean([p.x for p in puntos_grupo])
                cy = np.mean([p.y for p in puntos_grupo])
                
                # Escribir el número de cruces sobre el grupo
                ax.text(cx, cy, str(cantidad), color='black', fontsize=11, fontweight='bold',
                        ha='center', va='center', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=cmap(label), boxstyle='round,pad=0.3'))
                
                # Añadir a la leyenda
                ax.scatter([], [], color=cmap(label), marker='o', label=f'Grupo {label}: {cantidad} cruces')
                texto_global = f"N{nodo_nuevo_id}\n({cantidad})"
                ax_global.text(cx, cy, texto_global, color='black', fontsize=8, fontweight='bold',
                        ha='center', va='center', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor=cmap(label), boxstyle='round,pad=0.2'))
                
    # Ajustes visuales
    ax.set_title(f"Distribución de los cruces \n{fecha_sel}", fontsize=14)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    
    # Colocar la leyenda fuera del gráfico si hay muchos grupos para no tapar los flujos
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Zoom automático al área de los flujos
    if flujos_originales_nodo:
        minx = min([l.bounds[0] for l in flujos_originales_nodo])
        miny = min([l.bounds[1] for l in flujos_originales_nodo])
        maxx = max([l.bounds[2] for l in flujos_originales_nodo])
        maxy = max([l.bounds[3] for l in flujos_originales_nodo])
        
        margen_x = (maxx - minx) * 0.1
        margen_y = (maxy - miny) * 0.1
        ax.set_xlim(minx - margen_x, maxx + margen_x)
        ax.set_ylim(miny - margen_y, maxy + margen_y)

    plt.tight_layout()
    plt.show()
    
    # Reporte por consola
    print(f"-> Nodo {nodo_nuevo_id}: {len(puntos_cruce_reales)} cruces reales. Formando {n_clusters_cruces} sub-grupos de alta densidad.")

# ---> NUEVO: MOSTRAR EL GRÁFICO GLOBAL FUERA DEL BUCLE <---
print("\n--- GENERANDO MAPA GLOBAL DE TODOS LOS NODOS ---")

# Filtrar leyenda para no repetir 
handles, labels_leg = ax_global.get_legend_handles_labels()
unique_legend = dict(zip(labels_leg, handles))
if unique_legend:
    ax_global.legend(unique_legend.values(), unique_legend.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))

ax_global.grid(True, linestyle=':', alpha=0.5)

# Aplicar el zoom global basándonos en todos los datos recopilados
if global_bounds:
    minx = min([b[0] for b in global_bounds])
    miny = min([b[1] for b in global_bounds])
    maxx = max([b[2] for b in global_bounds])
    maxy = max([b[3] for b in global_bounds])
    
    margen_x = (maxx - minx) * 0.05
    margen_y = (maxy - miny) * 0.05
    ax_global.set_xlim(minx - margen_x, maxx + margen_x)
    ax_global.set_ylim(miny - margen_y, maxy + margen_y)

# Mostrar la figura global (una vez que los plots individuales hayan terminado)
# Pasamos la figura específica para no crear conflictos.

ax_global.set_title(f"Nodos de intersecciones detectados \n{fecha_sel}", fontsize=18, fontweight='bold', y=0.98)
ax_global.set_xlabel("Longitud")
ax_global.set_ylabel("Latitud")

# fig_global.tight_layout()
fig_global.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.82)
plt.figure(fig_global.number)
fig_global.show()

#%%
# ==================================================================================================================== #
# ======================= FILTRADO DE GRUPOS, ARISTAS Y POLÍGONOS (AÑADIDO POSTERIOR) ================================ #
# ==================================================================================================================== #

print("\n--- GENERANDO ARISTAS Y POLÍGONOS PARA GRUPOS FILTRADOS ---")

nodos_finales = []
# Parámetros del filtro estricto
peso_minimo_subgrupo = 100
umbral_cercania_subgrupo = 0.25 # Usamos el mismo umbral (0.25) que definiste arriba

for nodo_original, nodo_nuevo_id in mapeo_nodos.items():
    grupos_implicados = nodos_a_grupos[nodo_original]
    
    # 1. Recuperar la frontera del sector correspondiente a este nodo

    # 2. Re-extraer flujos y puntos (ya que no se guardaron globalmente en el bucle anterior)
    df_flujos_nodo = df_resultado_sectores[
        df_resultado_sectores['cluster_global'].isin(grupos_implicados)
    ]
    lineas_nodo = df_flujos_nodo['Line'].tolist()
    ids_nodo = df_flujos_nodo['Original_ID'].tolist() # Extraemos los IDs de vuelo
    
    puntos_cruce_reales = []
    ids_por_punto = []

    for i in range(len(lineas_nodo)):
        for j in range(i + 1, len(lineas_nodo)):
            if lineas_nodo[i].intersects(lineas_nodo[j]):
                inter = lineas_nodo[i].intersection(lineas_nodo[j])
                if inter.geom_type == 'Point':
                    puntos_cruce_reales.append(inter)
                    ids_por_punto.append((ids_nodo[i], ids_nodo[j])) # Relacionamos punto con flujos
                elif inter.geom_type == 'MultiPoint':
                    for p in inter.geoms:
                        puntos_cruce_reales.append(p)
                        ids_por_punto.append((ids_nodo[i], ids_nodo[j]))
                        
    # 3. Aplicar DBSCAN y Filtrar
    puntos_validos_geom = []
    labels_filtradas = []
    ids_validos_sab = set()
    n_clusters_validos = 0

    if len(puntos_cruce_reales) > 0:
        # 1. Crear un polígono que englobe todos los puntos de cruce reales
        # Usamos convex_hull para envolverlos y un pequeño buffer por si 
        # solo hay 2 puntos (línea) o están perfectamente alineados, dándole área real.
        poligono_conflicto = MultiPoint(puntos_cruce_reales).convex_hull.buffer(0.005)
            
        max_area_interseccion = 0
        sector_id_nodo = None
        frontera_interna_unida = None

        # 2. Evaluar qué sector abarca la mayor parte de este polígono
        for _, row_sector in gdf_sectores.iterrows():
            geometria_sector = row_sector.geometry 
            geometry_DGI = DF_info_conf[DF_info_conf['SECTOR_ID'] == 'LECMDGI']['Contorno Sector'].values[0]
            geometry_BLI = DF_info_conf[DF_info_conf['SECTOR_ID'] == 'LECMBLI']['Contorno Sector'].values[0]
            
                # Solo calculamos el área si hay intersección (para ahorrar tiempo de cómputo)
            if geometria_sector.intersects(poligono_conflicto):
                    # Calculamos el área de solapamiento
                area_interseccion = geometria_sector.intersection(poligono_conflicto).area    
                    # Nos quedamos con el sector que contenga la mayor área del conflicto
                if area_interseccion > max_area_interseccion:
                    max_area_interseccion = area_interseccion
                    sector_id_nodo = row_sector['SECTOR_ID']
                   
            # 3. Si encontramos el sector principal, extraemos su frontera interna
            if sector_id_nodo:
                lineas_sector = gdf_fronteras_internas[gdf_fronteras_internas['SECTOR_ID'] == sector_id_nodo]['geometry']
                frontera_interna_unida = lineas_sector.union_all()
        
        coords_cruces = np.array([[p.x, p.y] for p in puntos_cruce_reales])
        db_cruces = DBSCAN(eps=eps_cruces, min_samples=min_samples_cruces).fit(coords_cruces)
        labels_temp = list(db_cruces.labels_)
        
        etiquetas_unicas = set(labels_temp) - {-1}
        
        for label in etiquetas_unicas:
            indices_grupo = [i for i, l in enumerate(labels_temp) if l == label]
            puntos_grupo = [puntos_cruce_reales[i] for i in indices_grupo]
            peso_grupo = len(puntos_grupo)
            
            print(f"-> Nodo {nodo_nuevo_id}: Grupo {label} con {peso_grupo} cruces.")

            geometria_grupo = MultiPoint(puntos_grupo)
            distancia_grupo = geometria_grupo.distance(frontera_interna_unida) if frontera_interna_unida else 0
            
            # Condición de descarte: Menos de 100 cruces o demasiado lejos de la frontera
            if peso_grupo < peso_minimo_subgrupo or (frontera_interna_unida and distancia_grupo > umbral_cercania_subgrupo):
                for i in indices_grupo:
                    labels_temp[i] = -1 # Se convierte en ruido
            else:
                puntos_validos_geom.extend(puntos_grupo)
                for i in indices_grupo:
                    id1, id2 = ids_por_punto[i]
                    ids_validos_sab.add(id1)
                    ids_validos_sab.add(id2)
                
        labels_filtradas = labels_temp
        n_clusters_validos = len(set(labels_filtradas) - {-1})
        print(f"Los cluster válidos son: {n_clusters_validos}")

    # Si no han sobrevivido grupos a este filtro tan estricto, pasamos al siguiente nodo
    if n_clusters_validos == 0:
        print(f"-> Nodo {nodo_nuevo_id}: Ningún grupo superó el filtro (peso >= 100 y cercanía).")
        continue

    # 4. Creación del Polígono (Aristas) apoyado en la frontera
    poligono_final = None
    if n_clusters_validos > 0 and frontera_interna_unida:
        multipunto_validos = MultiPoint(puntos_validos_geom)
        
        # Buscamos la intersección con la frontera cercana expandiendo el área de influencia
        zona_influencia = multipunto_validos.buffer(umbral_cercania_subgrupo*1.25)
        segmento_frontera = frontera_interna_unida.intersection(zona_influencia)
        
        puntos_para_poligono = list(multipunto_validos.geoms)
        
        # Extraer los vértices de la frontera para unirlos a los puntos de cruce
        if not segmento_frontera.is_empty:
            if segmento_frontera.geom_type in ['LineString', 'LinearRing']:
                puntos_para_poligono.extend([Point(c) for c in segmento_frontera.coords])
            elif segmento_frontera.geom_type == 'MultiLineString':
                for line in segmento_frontera.geoms:
                    puntos_para_poligono.extend([Point(c) for c in line.coords])
                    
        # Envolvente convexa para crear las aristas lógicas
        if len(puntos_para_poligono) >= 3:
            poligono_final = MultiPoint(puntos_para_poligono).convex_hull
            # --- NUEVO: INTERSECCIÓN CON EL MALLADO ---
            # Verificamos qué celdas de gdf_mallado se solapan con el polígono final
            if poligono_final.is_valid and not poligono_final.is_empty:
                celdas_intersectadas = gdf_mallado[gdf_mallado.geometry.intersects(poligono_final)]
                
                # Extraemos el identificador de la celda. 
                # NOTA: Si tu gdf_mallado tiene una columna específica para el ID de celda (ej. 'Cell_ID'), 
                # cambia ".index.tolist()" por "['Cell_ID'].tolist()".
                celdas_del_poligono = celdas_intersectadas.index.tolist()

    # NUEVO: Filtrar el DataFrame original para quedarnos solo con la información de los flujos SAB válidos
    flujos_sab_finales = df_flujos_nodo[df_flujos_nodo['Original_ID'].isin(ids_validos_sab)][['Original_ID', 'Line']].to_dict('records')

    nodos_finales.append({
        'nodo_id': nodo_nuevo_id,
        'sector_id': sector_id_nodo,
        'peso': n_clusters_validos,
        'Flujos de SAB': flujos_sab_finales,
        'Celda': celdas_del_poligono,
        'forma de SAB': poligono_final
    })

    print(f"El nodo {nodo_nuevo_id} estáen el sector {sector_id_nodo}")
    # 5. Representación Gráfica del Polígono y Grupos Filtrados
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar flujos
    for linea in flujos_originales_nodo:
        ax.plot(*linea.xy, color='steelblue', alpha=0.15, linewidth=1.5)
        
    # Dibujar la frontera del sector afectada
    poly_sec = DF_info_conf[DF_info_conf['SECTOR_ID'] == sector_id_nodo]['Contorno Sector'].values[0]
    if poly_sec.geom_type == 'Polygon':
        x_poly, y_poly = poly_sec.exterior.xy
        ax.plot(x_poly, y_poly, color='black', linewidth=1.5, label='Frontera')
    elif poly_sec.geom_type == 'MultiPolygon':
        for p in poly_sec.geoms:
            x_poly, y_poly = p.exterior.xy
            ax.plot(x_poly, y_poly, color='black', linewidth=1.5, label='Frontera')
    
    # Dibujar los puntos de cruce (Solo los válidos resaltados)
    cmap = cm.get_cmap('tab20', max(1, n_clusters_validos))
    for idx, p in enumerate(puntos_cruce_reales):
        label = labels_filtradas[idx]
        if label == -1: 
            ax.scatter(p.x, p.y, color='gray', s=5, alpha=0.2, marker='x') # Los filtrados quedan muy suaves
        else: 
            ax.scatter(p.x, p.y, color=cmap(label), s=35, alpha=0.9, zorder=5, label='Puntos de cruce')
                
    # Etiquetas de cantidad
    for label in set(labels_filtradas) - {-1}:
        puntos_grupo = [puntos_cruce_reales[i] for i, l in enumerate(labels_filtradas) if l == label]
        cantidad = len(puntos_grupo)
        cx, cy = np.mean([p.x for p in puntos_grupo]), np.mean([p.y for p in puntos_grupo])
        
        # ax.text(cx, cy, str(cantidad), color='black', fontsize=12, fontweight='bold',
        #         ha='center', va='center', zorder=10,
        #         bbox=dict(facecolor='white', alpha=0.8, edgecolor=cmap(label), boxstyle='round,pad=0.3'))
        ax.scatter([], [], color=cmap(label), marker='o', label=f'Puntos de cruce')

    # Dibujar el Polígono de aristas envolventes
    if poligono_final and poligono_final.geom_type == 'Polygon':
        x_poly, y_poly = poligono_final.exterior.xy
        ax.plot(x_poly, y_poly, color='magenta', linewidth=3.5, linestyle='-', zorder=6, label='SAB')
        ax.fill(x_poly, y_poly, color='magenta', alpha=0.15, zorder=1)

    # Ajustes finales del gráfico
    ax.set_title(f"Resultado del diseño \n{fecha_sel}", fontsize=14)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Zoom
    if flujos_originales_nodo:
        minx = min([l.bounds[0] for l in flujos_originales_nodo])
        miny = min([l.bounds[1] for l in flujos_originales_nodo])
        maxx = max([l.bounds[2] for l in flujos_originales_nodo])
        maxy = max([l.bounds[3] for l in flujos_originales_nodo])
        margen_x = (maxx - minx) * 0.1
        margen_y = (maxy - miny) * 0.1
        ax.set_xlim(minx - margen_x, maxx + margen_x)
        ax.set_ylim(miny - margen_y, maxy + margen_y)

    plt.tight_layout()
    plt.show()
    
    print(f"-> Nodo {nodo_nuevo_id}: {n_clusters_validos} grupos críticos detectados. Polígono generado.")

df_nodos_finales = pd.DataFrame(nodos_finales)
df_nodos_finales.to_pickle(PATH_sabs + f'SAB_tipo2_{fecha_sel}.pkl')
df_nodos_finales.to_csv(PATH_sabs + f'SAB_tipo2_{fecha_sel}.csv', index=False)
print(f"\nArchivo de SABs tipo 2 guardado 'SAB_tipo2_{fecha_sel}.pkl'.")

#%% representación gráfica
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- REPRESENTACIÓN ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# representación del resultado final

fig, ax = plt.subplots(figsize=(14, 9))

# 1. Dibujar el mallado de fondo
gdf_mallado.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.6, zorder=1)

# 2. Dibujar contornos de sectores y sus etiquetas
for index, row in DF_info_conf.iterrows():
    poly = row['Contorno Sector']
    if poly.geom_type == 'Polygon':
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
    elif poly.geom_type == 'MultiPolygon':
        for p in poly.geoms:
            x, y = p.exterior.xy
            ax.fill(x, y, alpha=0.15, edgecolor='black', linewidth=1.5, zorder=2)
            
    ax.text(poly.centroid.x, poly.centroid.y, row['SECTOR_ID'], 
             fontsize=10, ha='center', weight='bold', alpha=0.6, zorder=3)

for nodo in nodos_finales:
    poly_sab = nodo['forma de SAB']
    
    # Validar que la geometría exista y no esté vacía por si acaso
    if poly_sab is not None and not poly_sab.is_empty:
        if poly_sab.geom_type == 'Polygon':
            x, y = poly_sab.exterior.xy
            ax.fill(x, y, facecolor='darkorange', edgecolor='saddlebrown', 
                    alpha=0.5, linewidth=1.5, zorder=4)
        elif poly_sab.geom_type == 'MultiPolygon':
            for p in poly_sab.geoms:
                x, y = p.exterior.xy
                ax.fill(x, y, facecolor='darkorange', edgecolor='saddlebrown', 
                        alpha=0.5, linewidth=1.5, zorder=4)

# Configurar el lienzo
ax.set_title(f"Resultados del 2º tipo de SAB \n {fecha_sel}", fontsize=16)
plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')

# Leyenda adaptada
poligono_leyenda_frontera = Patch(facecolor='darkorange', edgecolor='saddlebrown', alpha=0.5, label='Frontera (Filtrados)')
plt.legend(handles=[poligono_leyenda_frontera], loc='upper right')

# Guardar y mostrar
plt.savefig(f'resultados_sab2_{fecha_sel}.png', bbox_inches='tight')
plt.show()
plt.close()

# %%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- GUARDADO ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


