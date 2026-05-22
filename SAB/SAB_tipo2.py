#%% SABs TIPO 2
# Para calcular la complejidad en cada celda
# identificar zonas con poco tráfico -> sabs tipo 2 
# 
#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import datetime
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import pickle
import seaborn as sns
import ast
import time as tm
import networkx as nx
import gc
from datetime import datetime, time

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
from sortedcontainers import SortedDict
from scipy.stats import norm
from scipy import stats


## Rutas de los archivos
RUTA_COMPLEJIDAD = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Junio2022\\test\\' # Asegúrate de que esta ruta es correcta

RUTA_MALLADO = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Resultados analisis flujo celda\\' # Asegúrate de que esta ruta es correcta

PATH_sabs = 'F:\\Users\\Lai\\Datos\\3. bloque optimizacion\\Definición de sabs\\'


# Pedir el día por la terminal de VS Code
# dia_input = input("Introduce el día para la carpeta (ej. 01 15): ")
dia_input = '1'

# Crear el objeto de fecha y el nombre de la carpeta (formato YYYY-MM-DD)
# Enero
fecha_data = datetime(2022, 6, int(dia_input))
dia_input = dia_input.strip().zfill(2)
fecha_sel = f'2022-06-{dia_input}'

# Junio
# fecha_data = datetime(2022, 6, int(dia_input))
# dia_input = dia_input.strip().zfill(2)
# fecha_sel = f'2022-06-{dia_input}'

nombre_carpeta = f"RESULTADOS_{fecha_data:%Y-%m-%d}"

# 4. Unir la ruta base con la nueva carpeta
RUTA_COMPLEJIDAD_DIA = os.path.join(RUTA_COMPLEJIDAD, nombre_carpeta)

tamaño_celda = 20

#%% 
#---------------------------------------------------------------------
#---------------------Cargar datos de complejidad---------------------
#---------------------------------------------------------------------

# 1. Cargar los datos de complejidad para la hora seleccionada
#ver_grafica = input("¿Deseas visualizar la gráfica de una franja horaria específica? (s/n): ").lower()
ver_grafica = 'n'
if ver_grafica == 's':
    hora_inic = input("Introduce la franja horaria de los datos de complejidad (solamente la hora de comienzo ej. 01 06 12): ")
    hora_fin = str(int(hora_inic) + 1).zfill(2)
    hora_inic = hora_inic.strip().zfill(2)
    df_complejidad_hora = pd.read_csv(RUTA_COMPLEJIDAD_DIA + '\\' + f'Complejidad_por_hora_{fecha_sel}_{hora_inic}-{hora_fin}.csv', sep=';')
else:
    print("Omitiendo representación gráfica de complejidad en una franja horaria. Continuando con el resto de las representaciones...")

# 2. Cargar los datos de la suma de complejidad
df_complejidad_sum = pd.read_csv(RUTA_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Suma_{fecha_sel}_Completo.csv', sep=';')

# 3. Cargar los datos de la media de complejidad
df_complejidad_media = pd.read_csv(RUTA_COMPLEJIDAD_DIA + '\\' + f'Complejidad_Media_{fecha_sel}_Diaria.csv', sep=';')


#%% 
#---------------------------------------------------------------------
#----------------Preparar los datos y la geometría--------------------
#---------------------------------------------------------------------

# 1. Cargar el mallado
df_mallado = pd.read_csv(RUTA_MALLADO + 'Mallado_CNF5A_gdf_cells.csv', sep=';')

# 2. Preparar la geometría
# Convertimos la columna 'Polygon' de texto a objetos geométricos reales
df_mallado['geometry'] = df_mallado['Polygon'].apply(wkt.loads)
gdf_mallado = gpd.GeoDataFrame(df_mallado, geometry='geometry')

# 3. Unir ambos datasets
# Usamos 'Cell_Name' del mallado y 'Celda' de complejidad como claves
if ver_grafica == 's':
    gdf_final_hora = gdf_mallado.merge(df_complejidad_hora, left_on='Cell_Name', right_on='Celda')

gdf_final_sum = gdf_mallado.merge(df_complejidad_sum, left_on='Cell_Name', right_on='Celda')
gdf_final_media = gdf_mallado.merge(df_complejidad_media, left_on='Cell_Name', right_on='Celda')

#%% 
#--------------------------------------------------------------------
#-----------------------Celdas de frontera---------------------------
#--------------------------------------------------------------------

border_cells = pd.read_csv(RUTA_MALLADO + f'CNF5A_border_cells_{tamaño_celda}.csv', sep=';')

print("Asignación inicial (configuración real) basada en intersecciones:")
print(gdf_mallado[['Cell_Name','Sector']].head())

# Filtramos todas las celdas de frontera de una vez
gdf_fronteras = gdf_mallado[gdf_mallado['Cell_Name'].isin(border_cells['Cell_Name'])]

## Extraer las fronteras de los sectores
# Extraemos los bordes de todas las celdas
bordes_todos = gdf_mallado.copy()
bordes_todos['geometry'] = bordes_todos.boundary

# Creamos una capa que solo contenga las líneas que separan sectores
# Disolvemos por sector para obtener el contorno exterior de cada "bloque" de sector
gdf_limites_sectores = gdf_mallado.dissolve(by='Sector').boundary

#%%
#---------------------------------------------------------------------
#-------------------------Celdas atípicas-----------------------------
#---------------------------------------------------------------------

umbral = 1.5

# SABs para una franja horaria específica
if ver_grafica == 's':
    # 1. Preparación y Limpieza de Columnas
    df_hora = df_complejidad_sum.copy()
    df_hora.columns = df_hora.columns.str.strip()
    col_name = 'Complejidad_Total_Dia'

    # 2. Cálculo de Estadísticos y Z-Score
    mu, sigma = df_hora[col_name].mean(), df_hora[col_name].std()
    df_hora['z_score'] = (df_hora[col_name] - mu) / sigma

    # 3. Filtrado de Outliers en un solo paso
    # Filtramos por umbral Y por pertenencia a frontera simultáneamente
    
    outliers_hora = df_hora[
        ((df_hora['z_score'])> umbral) & 
        (df_hora['Celda'].isin(border_cells['Cell_Name']))
    ].sort_values(by=col_name, ascending=False)

    print("Celdas atípicas (SABs) para la suma de complejidad:")
    print(outliers_hora[['Celda', col_name, 'z_score']])
    # 4. Unión con Geometría (Mallado)
    gdf_sab_hora = gdf_mallado.merge(outliers_hora, left_on='Cell_Name', right_on='Celda')
else:
    print()

# SABs para la suma de complejidad
# 1. Preparación y Limpieza de Columnas
df_sum = df_complejidad_sum.copy()
df_sum.columns = df_sum.columns.str.strip()
col_name = 'Complejidad_Total_Dia'

# 2. Cálculo de Estadísticos y Z-Score
mu, sigma = df_sum[col_name].mean(), df_sum[col_name].std()
df_sum['z_score'] = (df_sum[col_name] - mu) / sigma

# 3. Filtrado de Outliers en un solo paso
# Filtramos por umbral Y por pertenencia a frontera simultáneamente

outliers_sum = df_sum[
    ((df_sum['z_score'])> umbral) & 
    (df_sum['Celda'].isin(border_cells['Cell_Name']))
].sort_values(by=col_name, ascending=False)

print("Celdas atípicas (SABs) para la suma de complejidad:")
print(outliers_sum[['Celda', col_name, 'z_score']])

# 4. Unión con Geometría (Mallado)
gdf_sab_sum = gdf_mallado.merge(outliers_sum, left_on='Cell_Name', right_on='Celda')
gdf_sab_sum.to_pickle(PATH_sabs + f'celdas_sab_tipo2_suma_{fecha_sel}.pkl')
gdf_sab_sum.to_csv(PATH_sabs + f'celdas_sab_tipo2_suma_{fecha_sel}.csv', index=False)

# SABs para la media de complejidad
# 1. Preparación y Limpieza de Columnas
df_media = df_complejidad_media.copy()
df_media.columns = df_media.columns.str.strip()
col_name = 'Media_Complejidad'

# 2. Cálculo de Estadísticos y Z-Score
mu, sigma = df_media[col_name].mean(), df_media[col_name].std()
df_media['z_score'] = (df_media[col_name] - mu) / sigma

# 3. Filtrado de Outliers en un solo paso
# Filtramos por umbral Y por pertenencia a frontera simultáneamente

outliers_media = df_media[
    ((df_media['z_score'])> umbral) & 
    (df_media['Celda'].isin(border_cells['Cell_Name']))
].sort_values(by=col_name, ascending=False)

print("Celdas atípicas (SABs) para la media de complejidad:")
print(outliers_media[['Celda', col_name, 'z_score']])

# 4. Unión con Geometría (Mallado)
gdf_sab_media = gdf_mallado.merge(outliers_media, left_on='Cell_Name', right_on='Celda')
gdf_sab_media.to_pickle(PATH_sabs + f'celdas_sab_tipo2_media_{fecha_sel}.pkl')
gdf_sab_media.to_csv(PATH_sabs + f'celdas_sab_tipo2_media_{fecha_sel}.csv', index=False)

#%%
#---------------------------------------------------------------------
#-----------------------Filtro de las celdas--------------------------
#---------------------------------------------------------------------

# Función para agrupar las celdas

# def agrupar_y_limitar_area_sabs(gdf_sab_celdas, gdf_mallado, col_complejidad):
#     """
#     Agrupa celdas SAB continuas, calcula el área de los sectores, 
#     y divide los SABs que superen el 25% del área media de un sector.
#     """
#     if gdf_sab_celdas.empty:
#         return gpd.GeoDataFrame()
        
#     # --- 1. CÁLCULO DEL ÁREA Y UMBRAL (25% del Sector Medio) ---
#     # Unimos todo el mallado por Sector para obtener los polígonos reales de los sectores
#     gdf_sectores = gdf_mallado.dissolve(by='Sector')
    
#     # Calculamos el área media de todos los sectores
#     area_media_sectores = gdf_sectores.geometry.area.mean()
#     umbral_area_sab = 0.25 * area_media_sectores
    
#     # --- 2. AGRUPACIÓN INICIAL DE CELDAS (NetworkX) ---
#     G = nx.Graph()
#     G.add_nodes_from(gdf_sab_celdas['Cell_Name'])
    
#     nombres = gdf_sab_celdas['Cell_Name'].tolist()
#     geometrias = gdf_sab_celdas['geometry'].tolist()
    
#     for i in range(len(nombres)):
#         for j in range(i + 1, len(nombres)):
#             if geometrias[i].touches(geometrias[j]):
#                 G.add_edge(nombres[i], nombres[j])
                
#     componentes = list(nx.connected_components(G))
#     gdf_sab = gdf_sab_celdas.copy()
#     gdf_sab['ID_SAB'] = -1 
    
#     # Filtro: Solo grupos de 2 o más celdas
#     id_grupo = 1
#     for grupo in componentes:
#         if len(grupo) >= 2:
#             gdf_sab.loc[gdf_sab['Cell_Name'].isin(grupo), 'ID_SAB'] = id_grupo
#             id_grupo += 1
            
#     gdf_sab = gdf_sab[gdf_sab['ID_SAB'] != -1].copy()
#     if gdf_sab.empty:
#         return gpd.GeoDataFrame()
        
#     # --- 3. RESTRICCIÓN DE ÁREA Y DIVISIÓN ---
#     # Columna para los IDs definitivos (ej. 1_1, 1_2 si se divide)
#     gdf_sab['ID_SAB_Final'] = gdf_sab['ID_SAB'].astype(str)
    
#     print(f"\n[Evaluación de Áreas] Media Sector: {area_media_sectores:.4f} | Umbral 25%: {umbral_area_sab:.4f}")
    
#     grupos = gdf_sab.groupby('ID_SAB')
#     for id_sab, grupo in grupos:
#         # Área total del SAB actual (sumando sus celdas)
#         area_sab = grupo.geometry.unary_union.area
        
#         if area_sab > umbral_area_sab:
#             # Calculamos en cuántos trozos hay que cortarlo
#             num_fragmentos = int(np.ceil(area_sab / umbral_area_sab))
#             print(f" -> SAB {id_sab} excede el umbral (Área: {area_sab:.4f}). Dividiendo en {num_fragmentos} fragmentos...")
            
#             # Para hacer un corte limpio, ordenamos las celdas a lo largo de su eje más largo
#             centroides = grupo.geometry.centroid
#             minx, miny, maxx, maxy = grupo.geometry.total_bounds
            
#             # Si es más ancho que alto, cortamos de Izquierda a Derecha (Longitud X)
#             if (maxx - minx) > (maxy - miny):
#                 indices_ordenados = centroides.x.sort_values().index
#             # Si es más alto que ancho, cortamos de Arriba a Abajo (Latitud Y)
#             else:
#                 indices_ordenados = centroides.y.sort_values().index
                
#             # Dividimos la lista de celdas en 'num_fragmentos' grupos iguales
#             chunks = np.array_split(indices_ordenados, num_fragmentos)
            
#             # Asignamos el nuevo sub-ID (ej. "3_1", "3_2", "3_3") a las celdas correspondientes
#             for i, chunk_indices in enumerate(chunks):
#                 gdf_sab.loc[chunk_indices, 'ID_SAB_Final'] = f"{id_sab}_{i+1}"
                
#     # --- 4. UNIFICACIÓN FINAL DE GEOMETRÍAS ---
#     columnas_utiles = ['ID_SAB_Final', col_complejidad, 'z_score', 'geometry']

#     print(gdf_sab)
#     gdf_entidades_final = gdf_sab[columnas_utiles].dissolve(by='ID_SAB_Final', aggfunc='mean').reset_index()
    
#     return gdf_entidades_final

def agrupar_y_limitar_area_sabs(gdf_sab_celdas, gdf_mallado, col_complejidad):
    """
    Agrupa celdas SAB continuas, limita su área y mantiene el registro 
    de qué celdas componen cada grupo final.
    """
    if gdf_sab_celdas.empty:
        print('gdf_sab empty')
        return gpd.GeoDataFrame()
    
    # Trabajamos sobre una copia para proteger el DataFrame original
    gdf_sab_celdas = gdf_sab_celdas.copy()
    
    # Si Cell_Name contiene listas, extraemos el contenido como texto.
    # Ej: ['Celda_1'] -> 'Celda_1' | ['C1', 'C2'] -> 'C1, C2'
    gdf_sab_celdas['Cell_Name'] = gdf_sab_celdas['Cell_Name'].apply(
        lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x
    )
    # Forzamos que toda la columna sea texto por seguridad
    gdf_sab_celdas['Cell_Name'] = gdf_sab_celdas['Cell_Name'].astype(str)

    # --- 1. CÁLCULO DEL ÁREA Y UMBRAL ---
    gdf_sectores = gdf_mallado.dissolve(by='Sector')
    area_media_sectores = gdf_sectores.geometry.area.mean()
    umbral_area_sab = 0.25 * area_media_sectores
    
    # --- 2. AGRUPACIÓN INICIAL (NetworkX) ---
    G = nx.Graph()
    G.add_nodes_from(gdf_sab_celdas['Cell_Name'])
    
    nombres = gdf_sab_celdas['Cell_Name'].tolist()
    geometrias = gdf_sab_celdas['geometry'].tolist()
    
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if geometrias[i].touches(geometrias[j]):
                G.add_edge(nombres[i], nombres[j])
                
    componentes = list(nx.connected_components(G))
    gdf_sab = gdf_sab_celdas.copy()
    gdf_sab['ID_SAB'] = -1 
    
    id_grupo = 1
    for grupo in componentes:
        if len(grupo) > 2:
            gdf_sab.loc[gdf_sab['Cell_Name'].isin(grupo), 'ID_SAB'] = id_grupo
            id_grupo += 1
            
    gdf_sab = gdf_sab[gdf_sab['ID_SAB'] != -1].copy()
    if gdf_sab.empty:
        return gpd.GeoDataFrame()
        
    # # --- 3. RESTRICCIÓN DE ÁREA Y DIVISIÓN ---
    gdf_sab['ID_SAB_Final'] = gdf_sab['ID_SAB'].astype(str)
    
    # grupos = gdf_sab.groupby('ID_SAB')
    # for id_sab, grupo in grupos:
    #     area_sab = grupo.geometry.unary_union.area
        
    #     if area_sab > umbral_area_sab:
    #         num_fragmentos = int(np.ceil(area_sab / umbral_area_sab))
    #         centroides = grupo.geometry.centroid
    #         minx, miny, maxx, maxy = grupo.geometry.total_bounds
            
    #         if (maxx - minx) > (maxy - miny):
    #             indices_ordenados = centroides.x.sort_values().index
    #         else:
    #             indices_ordenados = centroides.y.sort_values().index
                
    #         chunks = np.array_split(indices_ordenados, num_fragmentos)
    #         for i, chunk_indices in enumerate(chunks):
    #             gdf_sab.loc[chunk_indices, 'ID_SAB_Final'] = f"{id_sab}_{i+1}"
                
    # --- 4. UNIFICACIÓN FINAL CON LISTA DE CELDAS ---
    # Definimos cómo agregar cada columna:
    # - Cell_Name: Las unimos en una lista (o string)
    # - Complejidad/z_score: Calculamos la media
    # - Geometry: Dissolve automático (union)
    agg_dict = {
        'Cell_Name': lambda x: list(x), # Crea una lista de nombres de celdas
        col_complejidad: 'mean',
        'z_score': 'mean'
    }

    # Realizamos el dissolve
    # Nota: Cell_Name debe estar en el DataFrame para que agg_dict funcione
    gdf_entidades_final = gdf_sab.dissolve(by='ID_SAB_Final', aggfunc=agg_dict).reset_index()
    
    # Opcional: Crear una columna con el número de celdas que tiene el grupo
    gdf_entidades_final['Num_Celdas'] = gdf_entidades_final['Cell_Name'].apply(len)
    
    return gdf_entidades_final

# Función para detectar cuñas
def tiene_cuna(sab_poly, umbral_grados=90):
    """Evalúa si el polígono tiene una esquina aguda hacia adentro (cuña)."""
    poligonos = list(sab_poly.geoms) if sab_poly.geom_type == 'MultiPolygon' else [sab_poly]
    for poly in poligonos:
        coords = list(poly.exterior.coords)
        for i in range(len(coords) - 1):
            p_prev, p_curr, p_next = np.array(coords[i - 1]), np.array(coords[i]), np.array(coords[i + 1])
            v1, v2 = p_prev - p_curr, p_next - p_curr
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0: continue
            
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angulo = np.degrees(np.arccos(cos_theta))
            
            if angulo < umbral_grados:
                punto_medio = Point((p_prev + p_next) / 2.0)
                if not poly.contains(punto_medio): # Si el punto medio cae fuera, es una cuña
                    return True 
    return False

# =====================================================================
# --- APLICACIÓN DE FILTROS (SOBRESCRIBIENDO VARIABLES ORIGINALES) ---
# =====================================================================
print("\n--- PROCESANDO SABS (ÁREAS Y CUÑAS) ---")

# --- 1. Para la SUMA diaria ---
# Fíjate que le pasamos gdf_mallado como segundo parámetro
gdf_agrupado_sum = agrupar_y_limitar_area_sabs(gdf_sab_sum, gdf_mallado, 'Complejidad_Total_Dia')

if not gdf_agrupado_sum.empty:
    # Aplicar también el filtro de las cuñas que hicimos antes
    mascara_sum = gdf_agrupado_sum['geometry'].apply(lambda geom: tiene_cuna(geom))
    gdf_sab_sum_f = gdf_agrupado_sum[~mascara_sum].copy()
else:
    gdf_sab_sum_f = gpd.GeoDataFrame()

# gdf_sum = gdf_agrupado_sum.copy()
print(f"SABs Suma Diaria -> Válidos (Divididos por área y sin cuña): {len(gdf_sab_sum_f)}")

print(gdf_agrupado_sum)

df_filtro_sum = gdf_agrupado_sum.copy
# .explode() separa las listas en filas individuales y .unique() quita duplicados
celdas_validas = gdf_agrupado_sum['Cell_Name'].explode().unique()

# 2. Aplicamos el filtro (Asegúrate de que 'Cell_Name' es el nombre correcto en gdf_sab_sum)
df_filtro_sum = gdf_sab_sum[gdf_sab_sum['Cell_Name'].isin(celdas_validas)]
df_filtro_sum.to_pickle(PATH_sabs + f'celdas_sab_tipo2_suma_{fecha_sel}.pkl')
df_filtro_sum.to_csv(PATH_sabs + f'celdas_sab_tipo2_suma_{fecha_sel}.csv', index=False)

gdf_agrupado_sum.to_csv(PATH_sabs + '\\' + f'SABs_Suma_Diaria_agrupados_{fecha_sel}.csv', index=False, encoding='latin1')    
gdf_agrupado_sum.to_pickle(PATH_sabs + '\\' + f'SABs_Suma_Diaria_agrupados_{fecha_sel}.pkl')

# --- 2. Para la MEDIA diaria ---
gdf_agrupado_media = agrupar_y_limitar_area_sabs(gdf_sab_media, gdf_mallado, 'Media_Complejidad')

if not gdf_agrupado_media.empty:
    mascara_media = gdf_agrupado_media['geometry'].apply(lambda geom: tiene_cuna(geom))
    gdf_sab_media_f = gdf_agrupado_media[~mascara_media].copy()
else:
    gdf_sab_media_f = gpd.GeoDataFrame()

# gdf_media = gdf_agrupado_media.copy()
print(gdf_agrupado_media)
# (Haz lo mismo para la HORA si lo necesitas)

#%%
#---------------------------------------------------------------------
#----------------------Representación Gráfica-------------------------
#---------------------------------------------------------------------


### Complejidad por horas
if ver_grafica == 's':
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Dibujamos el mapa de calor
    gdf_final_hora.plot(column='Suma_Complejidad_total', 
                ax=ax, 
                cax=cax,
                legend=True,
                cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
                edgecolor='black', 
                linewidth=0.5,
                legend_kwds={'label': "Nivel de Complejidad"})

    plt.title(f'Representación de Complejidad por celdas\nFecha: {dia_input}-06-2022 | Franja: {hora_inic}-{hora_fin}h', 
            fontsize=14, 
            pad=20,          # Añade espacio entre el título y la gráfica
            fontweight='bold')    # Asegura que esté centrado arriba
    
    # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
    gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

    # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
    gdf_limites_sectores.plot(ax=ax, 
                          color='red', 
                          linewidth=2.5, 
                          zorder=3)
    
    
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Guardar o mostrar
    plt.show()
    # Representación de los SABs
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Dibujamos el mapa de calor
    gdf_final_hora.plot(column='Suma_Complejidad_total', 
                    ax=ax, 
                    cax=cax,
                    legend=True,
                    cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
                    edgecolor='black', 
                    linewidth=0.5,
                    legend_kwds={'label': "Nivel de Complejidad"})
        
    plt.title(f'Representación de Complejidad por celdas\nFecha: {dia_input}-06-2022| Franja: {hora_inic}-{hora_fin}h', 
                fontsize=14, 
                pad=20,          # Añade espacio entre el título y la gráfica
                fontweight='bold', 
                loc='center')    # Asegura que esté centrado arriba
        # # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
        # gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

        # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
    gdf_limites_sectores.plot(ax=ax, 
                            color='black', 
                            linewidth=2.0, 
                            zorder=3)
        
    gdf_sab_hora.plot(ax=ax, 
                    color='red', 
                    linewidth=2.5, 
                    zorder=3)

    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

        # Guardar o mostrar
    plt.show()
else:
    print()


#-----------------------------------------------------------------------------
# ### Complejidad total de un día entero
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)

# # Dibujamos el mapa de calor
# gdf_final_sum.plot(column='Complejidad_Total_Dia', 
#                ax=ax, 
#                cax=cax,
#                legend=True,
#                cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
#                edgecolor='black', 
#                linewidth=0.5,
#                legend_kwds={'label': "Nivel de Complejidad"})

# # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
# gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

# # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
# gdf_limites_sectores.plot(ax=ax, 
#                       color='red', 
#                       linewidth=2.5, 
#                       zorder=3)

# plt.title(f'Representación de la suma de Complejidad por Celda \nFecha: {fecha_sel}', 
#           fontsize=14, 
#           pad=20,          # Añade espacio entre el título y la gráfica
#           fontweight='bold', 
#           loc='center')    # Asegura que esté centrado arribaplt.xlabel('Longitud')
# plt.ylabel('Latitud')

# # Guardar o mostrar
# plt.show()

# 1. Configuración de la figura y el divisor para la barra de color
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1) # Barra de color más fina

# 2. Dibujamos el mapa de calor (Complejidad por Celda)
gdf_final_sum.plot(column='Complejidad_Total_Dia', 
                   ax=ax, 
                   cax=cax,
                   legend=True,
                   cmap='YlGnBu', 
                   edgecolor='black', 
                   linewidth=0.1,
                   legend_kwds={'label': "Nivel de Complejidad", 'orientation': "vertical"})

# 3. Dibujamos las capas de líneas
# Fronteras (Línea roja más fina)
gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0, zorder=5)

# Límites de sectores (Línea roja más gruesa)
gdf_limites_sectores.plot(ax=ax, color='red', linewidth=2.5, zorder=3)

# 4. Creación de la LEYENDA en la esquina superior derecha
custom_lines = [
    Line2D([0], [0], color='red', lw=1.0, label='Fronteras'),
    Line2D([0], [0], color='red', lw=2.5, label='Límites de Sectores')
]
ax.legend(handles=custom_lines, loc='upper right', fontsize='small', framealpha=0.9)

# 5. TÍTULOS Y EJES
# Usar ax.set_title asegura que el centro se calcule sobre el mapa, ignorando la barra de color
ax.set_title(f'Suma de Complejidad por Celda\nFecha: {dia_input}-06-2022', 
             fontsize=16, 
             fontweight='bold', 
             pad=15)

ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# # Representación de los SABs
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)

# # Dibujamos el mapa de calor
# gdf_final_sum.plot(column='Complejidad_Total_Dia', 
#                 ax=ax, 
#                 cax=cax,
#                 legend=True,
#                 cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
#                 edgecolor='black', 
#                 linewidth=0.0,
#                 legend_kwds={'label': "Nivel de Complejidad"})
    
# plt.title(f'Representación de Complejidad por celdas\nFecha: {fecha_sel}', 
#             fontsize=14, 
#             pad=20,          # Añade espacio entre el título y la gráfica
#             fontweight='bold', 
#             loc='center')    # Asegura que esté centrado arriba
#     # # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
#     # gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

#     # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
# gdf_limites_sectores.plot(ax=ax, 
#                           color='black', 
#                           linewidth=2.0, 
#                           zorder=3)
    
# gdf_sab_sum.plot(ax=ax, 
#                 color='red', 
#                 linewidth=2.5, 
#                 zorder=3)

# plt.xlabel('Longitud')
# plt.ylabel('Latitud')

#     # Guardar o mostrar
# plt.show()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1) # Reducido a 3% para mejor proporción

# 1. Dibujamos el mapa de calor de la complejidad
gdf_final_sum.plot(column='Complejidad_Total_Dia', 
                   ax=ax, 
                   cax=cax,
                   legend=True,
                   cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
                   edgecolor='black', 
                   linewidth=0.0,
                   legend_kwds={'label': "Nivel de Complejidad (Suma)", 'orientation': "vertical"})

# 2. Dibujamos las líneas de los límites de sectores
gdf_limites_sectores.plot(ax=ax, 
                          color='black', 
                          linewidth=2.0, 
                          zorder=3)
    
# 3. Dibujamos los SABs
# Nota: Al usar color='red', si son polígonos se rellenarán de rojo sólido. 
# Si solo quieres el contorno, cambia "color='red'" por "facecolor='none', edgecolor='red'"
# gdf_agrupado_sum.plot(ax=ax, 
#                  color='red', 
#                  linewidth=2.5, 
#                  zorder=4)

gdf_sab_sum.plot(ax=ax, 
                 color='red', 
                 linewidth=2.5, 
                 zorder=4)

# 4. Construcción de la LEYENDA en la esquina superior derecha
custom_lines = [
    Line2D([0], [0], color='black', lw=2.0, label='Límites de Sectores'),
    # Usamos Patch para que la leyenda muestre un cuadro rojo (si los SAB se rellenan). 
    # Si cambiaste los SAB a solo líneas de contorno, usa Line2D en su lugar.
    Patch(facecolor='red', edgecolor='red', label='Celdas SAB') 
]
ax.legend(handles=custom_lines, loc='upper right', fontsize='small', framealpha=0.9)

# 5. TÍTULOS Y EJES centrados en el mapa (ignorando el cax)
ax.set_title(f'Representación de Suma de Complejidad por Celdas\nFecha: {dia_input}-06-2022', 
             fontsize=16, 
             fontweight='bold', 
             pad=15)

ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------
### Complejidad media 
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)

# # Dibujamos el mapa de calor
# gdf_final_media.plot(column='Media_Complejidad', 
#                ax=ax, 
#                cax=cax,
#                legend=True,
#                cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
#                edgecolor='black', 
#                linewidth=0.5,
#                legend_kwds={'label': "Nivel de Complejidad"})

# # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
# gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

# # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
# gdf_limites_sectores.plot(ax=ax, 
#                           color='red', 
#                           linewidth=2.5, 
#                           zorder=3)

# plt.title(f'Representación de la media de Complejidad por Celda \nFecha: {fecha_sel}', 
#           fontsize=14, 
#           pad=20,          # Añade espacio entre el título y la gráfica
#           fontweight='bold', 
#           loc='center')    # Asegura que esté centrado arriba
# plt.xlabel('Longitud')
# plt.ylabel('Latitud')

# # Guardar o mostrar
# plt.show()

# # Representación de los SABs
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)

# # Dibujamos el mapa de calor
# gdf_final_media.plot(column='Media_Complejidad', 
#                 ax=ax, 
#                 cax=cax,
#                 legend=True,
#                 cmap='YlGnBu',  # Escala de color: Amarillo a Rojo
#                 edgecolor='black', 
#                 linewidth=0.5,
#                 legend_kwds={'label': "Nivel de Complejidad"})
    
# plt.title(f'Representación de Complejidad por celdas\nFecha: {fecha_sel}', 
#             fontsize=14, 
#             pad=20,          # Añade espacio entre el título y la gráfica
#             fontweight='bold', 
#             loc='center')    # Asegura que esté centrado arriba
#     # # 2. Dibujamos solo los bordes (facecolor='none') sobre el 'ax' existente
#     # gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=5)

#     # 3. Dibujamos las líneas de los límites de sectores (con un color y grosor diferente para destacarlos)
# gdf_limites_sectores.plot(ax=ax, 
#                           color='black', 
#                           linewidth=2.0, 
#                           zorder=3)
    
# gdf_sab_sum.plot(ax=ax, 
#                 color='red', 
#                 linewidth=2.5, 
#                 zorder=3)

# plt.xlabel('Longitud')
# plt.ylabel('Latitud')

#     # Guardar o mostrar
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

# 1. Dibujamos el mapa de calor (Media de Complejidad)
gdf_final_media.plot(column='Media_Complejidad', 
                     ax=ax, 
                     cax=cax,
                     legend=True,
                     cmap='YlGnBu', 
                     edgecolor='black', 
                     linewidth=0.1,  # Reducido para no saturar visualmente
                     legend_kwds={'label': "Nivel de Complejidad (Media)", 'orientation': "vertical"})

# 2. Dibujamos los límites de sectores y las celdas SAB
gdf_limites_sectores.plot(ax=ax, color='red', linewidth=2.5, zorder=3)

gdf_fronteras.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0, zorder=5)

# Usamos facecolor='none' si los SAB son polígonos y solo queremos el borde, 
# o simplemente las dibujamos si son líneas.
gdf_agrupado_sum.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=4)
# gdf_sab_sum.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=4)

# 3. Construimos la LEYENDA manual en la esquina superior derecha
custom_lines = [
    Line2D([0], [0], color='red', lw=2.5, label='Fronteras de Sectores'),
    Line2D([0], [0], color='red', lw=1.5, label='Celdas de frontera')
]
ax.legend(handles=custom_lines, loc='upper right', fontsize='small', framealpha=0.9)

# 4. TÍTULO Y EJES centrados en el mapa
ax.set_title(f'Media de Complejidad de 24h por Celdas\nFecha: {dia_input}-06-2022', 
             fontsize=16, fontweight='bold', pad=15)

ax.set_xlabel('Longitud [º]')
ax.set_ylabel('Latitud [º]')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Configuración principal de la figura
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Divisor para la barra de color (más fina y proporcionada)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

# 1. Dibujamos el mapa de calor (Media de Complejidad)
gdf_final_media.plot(column='Media_Complejidad', 
                     ax=ax, 
                     cax=cax,
                     legend=True,
                     cmap='YlGnBu',  
                     edgecolor='black', 
                     linewidth=0.1,  # Un grosor mínimo suele quedar mejor que 0.0 para mallas
                     legend_kwds={'label': "Nivel de Complejidad (Media)", 'orientation': "vertical"})

# 2. Dibujamos las líneas de los límites de sectores
gdf_limites_sectores.plot(ax=ax, 
                          color='black', 
                          linewidth=1.0, 
                          zorder=3)
    
# 3. Dibujamos los SABs
# Si gdf_sab_sum contiene polígonos, se rellenarán de rojo. 
# Si solo quieres el contorno rojo sin relleno, usa: facecolor='none', edgecolor='red'
# gdf_agrupado_media.plot(ax=ax, 
#                  color='red', 
#                  linewidth=2.5, 
#                  zorder=4)

gdf_sab_media.plot(ax=ax, 
                 color='red', 
                 linewidth=2.5, 
                 zorder=4)

# 4. Construcción de la LEYENDA (Esquina superior derecha)
custom_lines = [
    Patch(facecolor='red', edgecolor='red', label='Celdas identificadas') 
]
ax.legend(handles=custom_lines, loc='upper right', fontsize='small', framealpha=0.9)

# 5. TÍTULOS Y EJES centrados en el mapa
ax.set_title(f'Celdas de frontera identificadas\nFecha: {dia_input}-06-2022', 
             fontsize=16, 
             fontweight='bold', 
             pad=15)

ax.set_xlabel('Longitud [º]', fontsize=12)
ax.set_ylabel('Latitud [º]', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# Ajuste automático de márgenes y renderizado
plt.tight_layout()
plt.show()

# SAB ----------------------------------------------------------

fig, ax_4 = plt.subplots(1, 1, figsize=(15, 10))
divider = make_axes_locatable(ax_4)
cax_4 = divider.append_axes("right", size="3%", pad=0.1)

# 1. Fondo: Celdas normales con su complejidad suma
gdf_final_sum.plot(column='Complejidad_Total_Dia', 
                   ax=ax_4, 
                   cax=cax_4,
                   cmap='YlGnBu', 
                   edgecolor='black', 
                   linewidth=0.1,
                   legend=True,
                   legend_kwds={'label': "Nivel de Complejidad (Suma)", 'orientation': "vertical"})

# 2. Dibujamos las líneas de los límites de sectores (CORREGIDO a ax=ax_4)
gdf_limites_sectores.plot(ax=ax_4, color='black', linewidth=2.0, zorder=3)

# 3. DIBUJAMOS LAS ENTIDADES SAB AGRUPADAS
if not gdf_sab_sum.empty:
    gdf_sab_sum.plot(column='ID_SAB', 
                     ax=ax_4, 
                     cmap='Set1',     # Mapa de colores para diferenciar cada SAB
                     linewidth=2.5, 
                     edgecolor='red', # Resaltamos el borde exterior unificado
                     zorder=4,
                     alpha=0.9)
    
    # Añadimos las etiquetas de texto
    for idx, row in gdf_sab_sum.iterrows():
        punto = row['geometry'].representative_point()
        nombre_etiqueta = str(row['ID_SAB'])
        
        ax_4.text(punto.x, punto.y, nombre_etiqueta, 
                  color='black', fontsize=10, fontweight='bold',
                  ha='center', va='center', zorder=5,
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

# 4. Construimos la LEYENDA para SAB y Sectores
custom_lines_4 = [
    Line2D([0], [0], color='black', lw=2.0, label='Límites de Sectores'),
    Patch(facecolor='gray', edgecolor='red', lw=2.5, label='Entidades SAB Agrupadas')
]
ax_4.legend(handles=custom_lines_4, loc='upper right', fontsize='small', framealpha=0.9)

# 5. TÍTULO Y EJES centrados en el mapa
ax_4.set_title(f'Suma de Complejidad y Entidades SAB Agrupadas\nFecha: {dia_input}-06-2022', 
               fontsize=16, fontweight='bold', pad=15)

ax_4.set_xlabel('Longitud [º]')
ax_4.set_ylabel('Latitud [º]')
ax_4.set_aspect('equal')
ax_4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------
# Reemplaza tu bloque de "Representación de los SABs" con esto:

fig, ax_4 = plt.subplots(1, 1, figsize=(12, 10))
divider = make_axes_locatable(ax_4)
cax = divider.append_axes("right", size="5%", pad=0.1)

# Fondo: Celdas normales con su complejidad
gdf_final_sum.plot(column='Complejidad_Total_Dia', 
                ax=ax_4, 
                cax=cax,
                cmap='YlGnBu', 
                edgecolor='black', 
                linewidth=0.5,
                legend=True,
                legend_kwds={'label': "Nivel de Complejidad"})

# 3. Dibujamos las líneas de los límites de sectores
gdf_limites_sectores.plot(ax=ax, color='black', linewidth=2.0, zorder=3)
    
    # 4. DIBUJAMOS LAS ENTIDADES SAB AGRUPADAS
if not gdf_sab_sum.empty:
    gdf_sab_sum.plot(column='ID_SAB', 
                                   ax=ax_4, 
                                   cmap='Set1',     # Mapa de colores para diferenciar cada SAB
                                   linewidth=2.5, 
                                   edgecolor='red', # Resaltamos el borde exterior unificado
                                   zorder=4,
                                   alpha=0.9)
    # Añadimos las etiquetas de texto
    for idx, row in gdf_sab_sum.iterrows():
        # representative_point asegura que el texto se imprima dentro de la geometría
        punto = row['geometry'].representative_point()
            
        # El texto será el ID_SAB (por ejemplo "1_1", "2", etc.)
        nombre_etiqueta = str(row['ID_SAB'])
            
        ax_4.text(punto.x, punto.y, nombre_etiqueta, 
                    color='black', fontsize=10, fontweight='bold',
                    ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

plt.xlabel('Longitud')
plt.ylabel('Latitud')

# Guardar o mostrar
plt.show()

