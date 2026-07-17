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
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import seaborn as sns
import gc
import ast
import math
import pickle
import time as t

import itertools
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

import geopy.distance
from geopy.distance import geodesic
from shapely.wkt import loads
from shapely.geometry import Polygon, Point, LineString, box, MultiLineString, MultiPolygon
from shapely.ops import nearest_points
from shapely.ops import unary_union
from shapely.ops import split

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

# dia_sel = '13'
fecha_sel = f"2022-{mes_sel}-{dia_sel}"

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
    
# horas para el analisis manual---------------------
print(f"la fecha de estudio es {fecha_sel} y la franja horaria es de {hora_ini_str} a {hora_fin_str}")

# Extraemos el día del string formateado
nombre_carpeta = f"RESULTADOS_{fecha_sel}"  # Nombre de la carpeta para guardar resultados

# 4. Unir la ruta base con la nueva carpeta
PATH_COMPLEJIDAD_DIA = os.path.join(PATH_COMPLEJIDAD, nombre_carpeta)

# Tamaño de celda en nm
cell_size_nm = 20

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- IMPORTACIÓN DE DATASETS DEL ANÁLISIS A NIVEL CELDA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

### Cargar dataset de tráfico, mallado, etc-------------

# DATASET ANÁLISIS FLUJOS POR CELDA: qué flujos atraviesan cada celda
DF_cells = pd.read_pickle(PATH_resultados + f'dataset_flujos_por_celda.pkl')

# DATASET ANÁLISIS CELDAS POR FLUJO: qué celdas atraviesa cada flujo
DF_Flujos = pd.read_pickle(PATH_resultados + f'dataset_celdas_por_flujo.pkl')

# DATASET DE TRÁFICO REAL: qué vuelos reales hay en el día seleccionado
DF_Trafico = pd.read_pickle(PATH_TRAFICO + f'dataset_vuelos_reales_{fecha_sel}.pkl')

# DATASET DE CELDAS FRONTERIZAS: qué celdas están en la frontera entre sectores
df_border_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_border_cells_{cell_size_nm}.pkl')

# DATASET DE CELDAS DEL MALLADO: qué celdas hay en el mallado (con su geometría)
mallado_cells = pd.read_pickle(PATH_bordes + f'{configuracion_estudio}_DF_cells.pkl')
df_mallado = pd.read_pickle(PATH_mallado + f'Mallado_{configuracion_estudio}_gdf_cells.pkl')

gdf_mallado = gpd.GeoDataFrame(df_mallado, geometry='Polygon')
gdf_mallado = gdf_mallado.rename_geometry('geometry')

### Filtrado de datos--------------------------------------

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

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------ IMPORTACIÓN DE DATASETS DE LA COMPLEJIDAD ------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

### Complejidad de un día completo-----------------------------------
df_complejidad_sum = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Suma_{fecha_sel}_Completo.pkl')
df_complejidad_media = pd.read_pickle(PATH_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Media_{fecha_sel}_Diaria.pkl')

# Cambiar el nombre de la columna 'celda' a 'Cell_Name'
df_complejidad_sum = df_complejidad_sum.rename(columns={'Celda': 'Cell_Name'})
df_complejidad_media = df_complejidad_media.rename(columns={'Celda': 'Cell_Name'})

## Calcular el Z-score de la complejidad para cada celda-------------
# (Asumiendo que la columna de valores numéricos se llama 'Complejidad_Total_Dia')
media_comp = df_complejidad_sum['Complejidad_Total_Dia'].mean()
std_comp = df_complejidad_sum['Complejidad_Total_Dia'].std()
df_complejidad_sum['z_score'] = (df_complejidad_sum['Complejidad_Total_Dia'] - media_comp) / std_comp

media_comp_m = df_complejidad_media['Media_Complejidad'].mean()
std_comp_m = df_complejidad_media['Media_Complejidad'].std()
df_complejidad_media['z_score'] = (df_complejidad_media['Media_Complejidad'] - media_comp_m) / std_comp_m
#--------------------------------------------------------------------

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
    if dia_horas == 'hora':
        df_final.to_csv(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}_{h_ini_str}_{h_fin_str}.csv",index=False)
        df_final.to_pickle(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}_{h_ini_str}_{h_fin_str}.pkl")
    elif dia_horas == 'dia':
        df_final.to_csv(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}.csv",index=False)
        df_final.to_pickle(PATH_resultados+ f"Matriz_complejidad_celdas_{fecha_sel}.pkl")

    df_comp_hora_med = df_final.copy()
else:
    print(f"No se encontraron archivos de complejidad para el rango {hora_inicio_busqueda}-{hora_fin_busqueda} en el día {fecha_sel}.")
#--------------------------------------------------------------------

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

print(f"Latitud mínima: {min_lat}, Latitud máxima: {max_lat}, Longitud mínima: {min_lon}, Longitud máxima: {max_lon}")

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- REPRESENTACIÓN DE LAS TRAYECTORIAS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

## -------------------------------------------------------------------------------------------------------------------- #
gdf_cells = gpd.GeoDataFrame(mallado_cells.copy(), geometry='Polygon')
gdf_cells = gdf_cells.set_geometry('Polygon')

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

# 5. GRAFICAR FLUJOS (Capa superior)
gdf_flujos.plot(ax=ax_3, color='red', linewidth=1, alpha=0.4, zorder=3)

# Añadir texto para los flujos
for idx, row in gdf_flujos.iterrows():
    # Calculamos el centroide de la geometría para ubicar el texto
    centroide = row['Line'].centroid
    
    # Reemplaza 'nombre_columna' por la columna de tu GeoDataFrame que tiene el texto (ej. 'ID', 'Flujo', 'Volumen')
    texto = str(row['Flujo_Clusterizado'])  # Ajusta según la columna que quieras mostrar
    
    ax_3.text(
        centroide.x, 
        centroide.y, 
        texto, 
        fontsize=9, 
        color='black', 
        weight='bold',
        ha='center', # Alineación horizontal
        va='center', # Alineación vertical
        zorder=4     # Por encima de la línea del flujo
    )
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
ax_3.set_title(f'Representación de todos los flujos sobre el mallado\nFecha: {fecha_sel}', 
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

### --- A. CONFIGURACIÓN DE PARÁMETROS ---
# Para detección de flujos
DISTANCIA_PROXIMIDAD = 0.2   # Grados (aprox 5-6 NM) para considerar "cerca"
TOLERANCIA_ANGULO = 5      # Grados de diferencia máxima para considerar "paralelo"
# Para filtro de flujos
DISTANCIA_EXTREMO = 0.01  # Distancia para considerar que están "unidos" (~1.2 NM)
MAX_INTERSECCIONES = 0  # Límite máximo de cruces permitidos (0, 1 o 2)
MAX_CONEXIONES_EXTREMOS = 10  # Límite máximo de uniones en extremos permitidas

### --- B. DETECCIÓN DE FLUJOS ---
# 1. Obtener las fronteras entre sectores --------------------------------------
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

# 2. Parámetros de filtrado -----------------------------------

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

# 3. Buscar flujos cercanos y paralelos --------------------------------------
buffer_fronteras = fronteras_totales.buffer(DISTANCIA_PROXIMIDAD)
sab_tipo1_flows = []
fronteras_cercanas = []

for idx, flujo in DF_Flujos.iterrows():
    linea_flujo = flujo['Line']
    nombre_flujo = flujo['Flujo_Clusterizado']

    print(f"Evaluando flujo: {nombre_flujo}")
    # ¿Está cerca de alguna frontera?
    if linea_flujo.intersects(buffer_fronteras):
        # Calculamos ángulos para verificar paralelismo
        angulo_flujo = calcular_angulo_linea(linea_flujo)
        
        # Comparamos con las fronteras cercanas
        es_paralelo = False
        frontera_detectada = None

        for frontera in fronteras_internas:
            print(f"Distancia: {linea_flujo.distance(frontera)}")

            if linea_flujo.distance(frontera) < DISTANCIA_PROXIMIDAD:
                # La función ahora ya maneja MultiLineString internamente
                angulo_front = calcular_angulo_linea(frontera)
                print(f"Ángulo flujo: {angulo_flujo}, Ángulo frontera: {angulo_front}")
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

sab_1_plot = sab_tipo1_flows.copy()

# Extraer todas las trayectorias del día para comparar
geometrias_dia = DF_Flujos['Line'].tolist()
nombres_dia = DF_Flujos['Flujo_Clusterizado'].tolist()
geometria_sab = [DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == f, 'Line'].iloc[0] for f in sab_tipo1_flows]

# extra. Extraer SOLO las trayectorias SAB identificadas (para el Criterio 1 - Cruces entre sí)
geometrias_sab = []
nombres_sab = []
for nombre_sab in sab_tipo1_flows:
    linea = DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == nombre_sab, 'Line'].iloc[0]
    geometrias_sab.append(linea)
    nombres_sab.append(nombre_sab)

### --- C. FILTRADO DE FLUJOS SAB TIPO 1 ---
sab_finales_criterios = []

for nombre_f in sab_tipo1_flows:
    # Obtener la geometría del flujo SAB evaluado
    linea_evaluada = DF_Flujos.loc[DF_Flujos['Flujo_Clusterizado'] == nombre_f, 'Line'].iloc[0]
    p_inicio = Point(linea_evaluada.coords[0])
    p_final = Point(linea_evaluada.coords[-1])
    
    cruces_entre_sabs = 0
    flujos_sab_intersecados = [] 
    flujos_unidos_dia = []       

    # # --- CRITERIO 1: Intersecciones SOLAMENTE ENTRE FLUJOS SABs ---
    # for nombre_otro_sab, linea_otro_sab in zip(nombres_sab, geometrias_sab):
    #     if nombre_f == nombre_otro_sab: # No ev0aluarlo contra sí mismo
    #         continue
            
    #     interseccion = linea_evaluada.intersection(linea_otro_sab)
    #     if not interseccion.is_empty:
    #         puntos_cruce = 0
    #         if isinstance(interseccion, Point):
    #             puntos_cruce = 1
    #         elif hasattr(interseccion, 'geoms'):
    #             puntos_cruce = len([g for g in interseccion.geoms if isinstance(g, Point)])
            
    #         if puntos_cruce > 0:
    #             cruces_entre_sabs += puntos_cruce
    #             if nombre_otro_sab not in flujos_sab_intersecados:
    #                 flujos_sab_intersecados.append(nombre_otro_sab)

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

### --- D. ACTUALIZACIÓN DE LAS CELDAS SAB TIPO 1 ---

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
if dia_horas == 'hora':
    df_celdas_sab_1.to_csv(PATH_sabs + f'celdas_sab_tipo1_{fecha_sel}_{hora_ini_str}_{hora_fin_str}.csv', index=False, encoding='latin1')
    df_celdas_sab_1.to_pickle(PATH_sabs + f'celdas_sab_tipo1_{fecha_sel}_{hora_ini_str}_{hora_fin_str}.pkl')
elif dia_horas == 'dia':
    df_celdas_sab_1.to_csv(PATH_sabs + f'celdas_sab_tipo1_{fecha_sel}.csv', index=False, encoding='latin1')
    df_celdas_sab_1.to_pickle(PATH_sabs + f'celdas_sab_tipo1_{fecha_sel}.pkl')

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
ax_res.set_title(f'Flujos paralelos identificados\nFecha: {fecha_sel}', fontsize=14)
ax_res.set_xlabel('Longitud [º]')
ax_res.set_ylabel('Latitud [º]')

# Crear una leyenda personalizada
patch_sab = mpatches.Patch(color='yellow', alpha=0.5, label='Celdas que atraviesan los flujos')
line_flow = Line2D([0], [0], color='red', linewidth=2.5, label='Flujos Paralelos Detectados')
line_context = Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.4, label='Otros flujos del día')

ax_res.legend(handles=[patch_sab, line_flow, line_context], loc='upper right', frameon=True)

plt.tight_layout()
plt.show()


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

# 5. Dibujar los FLUJOS PARALELOS detectados (en Rojo Intenso)
for idx, flujo in DF_Flujos.iterrows():
    nombre_flujo = flujo['Flujo_Clusterizado']
    x_f, y_f = flujo['Line'].xy
    
    if nombre_flujo in sab_tipo1_flows:
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
ax_res.set_title(f'Flujos paralelos identificados\nFecha: {fecha_sel}', fontsize=14)
ax_res.set_xlabel('Longitud [º]')
ax_res.set_ylabel('Latitud [º]')

# Crear una leyenda personalizada
line_flow = Line2D([0], [0], color='red', linewidth=2.5, label='Flujos Paralelos Detectados')
line_context = Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.4, label='Otros flujos del día')

ax_res.legend(handles=[line_flow, line_context], loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

# %%
#%%
# ------------------------------------------------------------------------------------ #
# -------------------------------- EVALUACIÓN DE SABS -------------------------------- #
# ------------------------------------------------------------------------------------ #

# OBJETIVO: se evalua las celdas del tipo 1 que tiene un nivel alto de complejidad

### --- A. CONFIGURACIÓN DE PARÁMETROS ---
# Definimos un umbral, por ejemplo: 10.0
UMBRAL = 2

### --- B. FUNCIONES PARA LA EVALUACIÓN DE SAB ---

# Función para evaluar las celdas del tipo 1 por su complejidad
def filtrar_sabs_por_complejidad(lista_evaluar, df_complejidad, umbral_maximo):
    """
    Filtra las celdas de Tipo 1. Si superan la complejidad máxima, se descartan.
    
    Args:
        lista_evaluar (list): Lista de IDs de celdas (ej. ['Cell_77', 'Cell_200'])
        df_complejidad (DataFrame): El DataFrame que contiene la columna 'Media_Complejidad'
        umbral_maximo (float): El valor límite de complejidad permitido.
        
    Returns:
        list: Solo las celdas que pasaron la evaluación (complejidad <= umbral).
    """
    # 1. Filtramos el DataFrame original para quedarnos solo con las celdas a evaluar
    # Nota: Usamos 'Cell_Name' o la columna correspondiente donde estén los IDs
    df_pendientes = df_complejidad[df_complejidad['Cell_Name'].isin(lista_evaluar)]
    print(f"La complejidad de las celdas a evaluar es:\n{df_pendientes[['Cell_Name', 'Media_Complejidad']].to_string(index=False)}")
    
    # 2. Aplicamos la lógica: se mantienen si complejidad <= umbral_maximo
    sabs_validados = df_pendientes[df_pendientes['Media_Complejidad'] <= umbral_maximo]
    # print(f"La complejidad del grupo es: {df_pendientes['Media_Complejidad'].sum()} y el umbral es: {umbral_maximo}.")
    # 3. Retornamos la lista de los que sí cumplen
    return list(sabs_validados['Cell_Name'])

# Función para que una vez identificadas las celdas inválidas, se identique los grupos 
# que contienen esas celdas descartadas

def grupo_es_valido(lista_celdas_grupo, celdas_permitidas):
    # Convertimos la lista de la fila a set para una operación rápida
    # Si la intersección NO está vacía, el grupo contiene celdas válidas
    return not set(lista_celdas_grupo).isdisjoint(celdas_permitidas)

## Proceso de evaluación-------------------------------------------------------------
sab_tipo1 = df_celdas_sab_1.copy()

# 1. Identificar celdas iniciales y coincidencias
tipo1_cells = set(sab_tipo1['Cell_Name'])

# 2. Evaluación !! Tipo 1 !! por Complejidad
evaluar_tipo1 = list(tipo1_cells)
# UMBRAL definido previamente (ej: 30.0)
tipo1_validadas = filtrar_sabs_por_complejidad(evaluar_tipo1, df_comp_hora_med, UMBRAL)

# Identificar flujos a eliminar (por pasar por celdas T1 complejas)
tipo1_descartadas = set(evaluar_tipo1) - set(tipo1_validadas)
flujos_a_eliminar = sab_tipo1[sab_tipo1['Cell_Name'].isin(tipo1_descartadas)]['Flujo'].unique()
print(f"Flujos a eliminar: {flujos_a_eliminar}")

# Filtrar SAB Tipo 1 final (usamos .copy() para evitar SettingWithCopyWarning)
sab1_final = sab_tipo1[~sab_tipo1['Flujo'].isin(flujos_a_eliminar)].copy()
flujo_localizados = sab1_final['Flujo'].unique()
celdas_t1_vivas = set(sab1_final['Cell_Name'])

if dia_horas == 'hora':
    sab1_final.to_pickle(PATH_sabs + f'SAB_tipo1_final_2022-{fecha_sel}_{hora_ini_str}-{hora_fin_str}.pkl')
    sab1_final.to_csv(PATH_sabs + f'SAB_tipo1_final_2022-{fecha_sel}_{hora_ini_str}-{hora_fin_str}.csv', index=False)
elif dia_horas == 'dia':
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


### --- EXTRA. Agrupa los flujos en caso de que ha detectado varios flujos para el mismo sab ---
# Sirve para el posterior diseño de formas porque si existe varios flujos para el mismo sab,
# el program realiza varias veces el mismo proceso de diseo.

# analizar los flujos para ver si son coincidentes
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
#----------------------------------------------------------------------------------------

### VIASUALIZACIÓN DE FLUJOS TIPO 1 Y CELDAS SAB TIPO 1 ---------------------------------
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

for _, celda in gdf_cells.iterrows():
    if celda['Cell_Name'] in celdas_t1_vivas:
        x, y = celda['Polygon'].exterior.xy
        ax_res.fill(x, y, color='yellow', alpha=0.5, edgecolor='orange', linewidth=1, zorder=3)

# 5. Dibujar los FLUJOS PARALELOS detectados (en Rojo Intenso)
for idx, flujo in DF_Flujos.iterrows():
    nombre_flujo = flujo['Flujo_Clusterizado']
    x_f, y_f = flujo['Line'].xy
    
    if nombre_flujo in flujo_localizados:
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
ax_res.set_xlim(-13.5, 0.0041667)
ax_res.set_ylim(40.394167, 45.5)
ax_res.set_aspect('equal')
ax_res.set_title(f'Flujos paralelos identificados\nFecha: {fecha_sel} {hora_ini_str}h-{hora_fin_str}h', fontsize=14)
ax_res.set_xlabel('Longitud [º]')
ax_res.set_ylabel('Latitud [º]')

# Crear una leyenda personalizada
line_flow = Line2D([0], [0], color='red', linewidth=2.5, label='Flujos Paralelos Detectados')
line_context = Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.4, label='Otros flujos del día')

ax_res.legend(handles=[line_flow, line_context], loc='upper right', frameon=True)

plt.tight_layout()
plt.show()


#%%
#-------------------------------------------------------------------------------- #
#------------------Diseño de las fronteras de sabs------------------------------- #
#-------------------------------------------------------------------------------- #

## Para los del tipo 1 -----------------------------------------------------------
# LOGICA: 
# 1. generar una línea paralela al flujo que se aleje del flujo con la misma distancia 
# que tiene el flujo con la frontera de referencia
# 2. extender esa línea para que corte el sector entero, y será la línea de corte para 
# dividir el sector en dos partes: el que contiene al flujo y el que no
# 3. La parte que contiene al flujo será el nuevo sab del tipo 1

print('1')
### --- A. Funcion para obtener la linea de corte ---
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
        max_longitud_contenida = 0.0

        for _, s_row in gdf_acc.iterrows():
            # Calculamos la geometría de la parte del flujo que queda dentro del sector
            linea_interseccion = s_row['Contorno Sector'].intersection(linea_flujo)
            
            # Si hay intersección y es de tipo lineal (LineString/MultiLineString), medimos su longitud
            if not linea_interseccion.is_empty:
                longitud_actual = linea_interseccion.length
                
                # Si este sector contiene más parte del flujo que los anteriores, lo guardamos como candidato
                if longitud_actual > max_longitud_contenida:
                    max_longitud_contenida = longitud_actual
                    sector_correspondiente = s_row
        
        # Umbral opcional: Puedes añadir un 'and max_longitud_contenida > 0' por seguridad
        if sector_correspondiente is None or max_longitud_contenida == 0:
            print(f"Aviso: El flujo {nombre_flujo} no está englobado ni interseca significativamente con ningún sector.")
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

if dia_horas =='hora':
    df_sabs_t1_final.to_pickle(PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}_{hora_ini_str}-{hora_fin_str}.pkl')
    df_sabs_t1_final.to_csv(PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}_{hora_ini_str}-{hora_fin_str}.csv', index=False)
elif dia_horas == 'dia':
    df_sabs_t1_final.to_pickle(PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}.pkl')
    df_sabs_t1_final.to_csv(PATH_sabs + f'SAB_tipo1_definitivo_{fecha_sel}.csv', index=False)

print(f"Se han generado {len(df_sabs_t1_final)} SABs definitivos de Tipo 1.")

# Visualización rápida para verificar
if not df_sabs_t1_final.empty:
    ax = df_sabs_t1_final.plot(color='cyan', alpha=0.5, edgecolor='blue', figsize=(10,10))
    plt.title("Geometrías Finales SABs Tipo 1")
    plt.show()

#%%

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
    df_sabs_t1_final.plot(ax=ax6, color='cyan', alpha=1, edgecolor='blue')


# 4. Crear LEYENDA MANUAL
custom_lines_6 = [
    Patch(facecolor='cyan', edgecolor='blue', alpha=1, label='Forma de SAB del tipo 1')
]
ax6.legend(handles=custom_lines_6, loc='upper right', framealpha=0.9)

# 5. Configuración final
ax6.set_title(f"Rediseño de fronteras de los SAB del tipo 1\nFecha: {fecha_sel} {hora_ini_str}h-{hora_fin_str}h")
ax6.set_xlabel('Longitud [º]')
ax6.set_ylabel('Latitud [º]')
ax6.set_aspect('equal')

plt.tight_layout()
plt.show()
