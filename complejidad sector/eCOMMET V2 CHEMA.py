#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- LIBRERÍAS QUE NECESITA EL CÓDIGO ----------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from datetime import datetime, time
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import ast
import time as tm
from sortedcontainers import SortedDict
import warnings
import joblib
from joblib import dump
from joblib import load

warnings.filterwarnings("ignore")

##### Registro del inicio del tiempo
inicio_tiempo = tm.time()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ IMPORTACIÓN DE BASE DE DATOS -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# CARACTERÍSTICAS DE LOS SECTORES Y SELECCION DEL TIPO DE TRÁFICO PARA EL CUAL SE DESEA CALCULAR LA COMPLEJIDAD
tipo_trafico = input('Seleccionar el tipo de tráfico para el cual se desea calcular la complejidad de los sectores de estudio (real/predicciones): ')

sectores_especificos = ['LECMASU', 'LECMBLL', 'LECMBLU', 'LECMDGL', 'LECMDGU', 'LECMPAL', 'LECMPAU', 'LECMSAO', 'LECMSAS']
niveles_vuelo_sectores = {
    'LECMASU' : [345, 660, 315], # Límite inferior, límite superior, diferencia FL
    'LECMBLL' : [0, 345, 345],
    'LECMBLU' : [345, 660, 315],
    'LECMDGL' : [0, 345, 345],
    'LECMDGU' : [345, 660, 315],
    'LECMPAL' : [0, 345, 345],
    'LECMPAU' : [345, 660, 315],
    'LECMSAO' : [0, 660, 660],
    'LECMSAS' : [0, 345, 345],
}


# DIRECTORIOS
PATH_ENTRADA = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\Junio2022\\'
PATH_INTERACCIONES = 'F:\\Users\\Lai\\Original\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\'
if tipo_trafico == 'real':
    PATH_RESULTADOS = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\RESULTADOS COMPLEJIDAD\\REAL\\Junio2022\\'
elif tipo_trafico == 'predicciones':
    PATH_RESULTADOS = 'F:\\Users\\Lai\\Datos\\2. bloque complejidad\\Datos\\RESULTADOS COMPLEJIDAD\\PREDICCIONES\\Junio2022\\'

# Pedir el día por la terminal de VS Code
dia_input = input("Introduce el día para la carpeta (ej. 15): ")

# Crear el objeto de fecha y el nombre de la carpeta (formato YYYY-MM-DD)
fecha_data = datetime(2022, 6, int(dia_input))
fecha_sel = f'2022-06-{dia_input}'

nombre_carpeta = f"RESULTADOS_{fecha_data:%Y-%m-%d}"

# 4. Unir la ruta base con la nueva carpeta
PATH_RESULTADOS_DIA = os.path.join(PATH_RESULTADOS, nombre_carpeta)

# 5. ¡MAGIA! VS Code crea la carpeta si no existe
if not os.path.exists(PATH_RESULTADOS_DIA):
    os.makedirs(PATH_RESULTADOS_DIA)
    print(f"Carpeta creada: {PATH_RESULTADOS_DIA}")

# IMPORTACIÓN DE LA BASE DE DATOS DE TRÁFICO CORRESPONDIENTE
if tipo_trafico == 'real':
    Trafico = pd.read_pickle(PATH_ENTRADA + f'dataset_vuelos_reales_{fecha_sel}.pkl')
elif tipo_trafico == 'predicciones':
    Trafico = pd.read_pickle(PATH_ENTRADA + f'dataset_vuelos_predicciones_{fecha_sel}.pkl')


# MATRIZ DE INTERACCIÓN DE FLUJOS
matriz_interacciones_flujos = pd.read_pickle(PATH_INTERACCIONES + 'Matriz_Interaccion_Flujos.pkl')

matriz_interacciones_flujos = pd.read_csv(
    PATH_flujos + "flow_trend_DF.csv",
    sep=";",
    encoding="latin1",
    dtype=None,        # intenta inferir tipos
    parse_dates=True,  # intenta convertir fechas
    low_memory=False
)


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- CREACIÓN DEL DATASET PARA CALCULAR LA COMPLEJIDAD --------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# MODIFICACIONES AL DATASET INICIAL DE TRÁFICO

# Renombrar ciertas columnas del dataset de tráfico
Trafico.rename(columns={'flightKey': 'flightID'}, inplace=True)
Trafico.rename(columns={'sectorEntryInstant': 'EntryTime'}, inplace=True)
Trafico.rename(columns={'sectorExitInstant': 'ExitTime'}, inplace=True)
Trafico.rename(columns={'modoCIN': 'EntryFL'}, inplace=True)
Trafico.rename(columns={'modoCOUT': 'ExitFL'}, inplace=True)
Trafico.rename(columns={'attitudIN': 'EntryTrend'}, inplace=True)
Trafico.rename(columns={'attitudIN-encoded': 'EntryTrend-encoded'}, inplace=True)
Trafico.rename(columns={'attitudOUT': 'ExitTrend'}, inplace=True)
Trafico.rename(columns={'attitudOUT-enconded': 'ExitTrend-enconded'}, inplace=True)
Trafico.rename(columns={'Flujo_Clusterizado': 'assignedFlow'}, inplace=True)
Trafico.rename(columns={'Clave_Flujo': 'assignedFlow_Key'}, inplace=True)

# Modificar el valor de la columna flightID para poder obtener aeronaves en un período de tiempo
Trafico['flightID'] = Trafico['flightID'].astype(str)
Trafico['flightID_original'] = Trafico['flightID']
Trafico['Sector'] = Trafico['Sector'].astype(str)
Trafico['flightID'] = Trafico['flightID'] + '_' + Trafico['Sector']

# Convertir las columnas EntryTime y ExitTime a tipo datetime para asegurar el formato
Trafico['EntryTime'] = pd.to_datetime(Trafico['EntryTime'])
Trafico['ExitTime'] = pd.to_datetime(Trafico['ExitTime'])


# CREACIÓN DE UN DATASET EN EL QUE SE AGRUPEN LAS AERONAVES QUE PASAN POR ALGÚN SECTOR DEL ACC EN PERÍODOS DE 5 MINUTOS



# Definir el período de estudio


start_date = f'2022-06-{dia_input} 00:00:00'
end_date = f'2022-06-{dia_input} 23:59:59'

# Crear un rango de tiempo para el período de estudio, en intervalos de 1 minuto
rango_tiempo = pd.date_range(start=start_date, end=end_date, freq='30T')

# Crear un diccionario para almacenar las aeronaves que pasan durante cada período de 5 minutos
aeronaves_por_periodo = {}

# Iterar sobre cada período de 5 minutos con un desplazamiento de 1 minuto
for inicio_periodo in rango_tiempo:
    fin_periodo = inicio_periodo + pd.Timedelta(minutes=60)
    # Filtrar las aeronaves que pasan durante el período de 5 minutos
    aeronaves_pasadas = Trafico[~((Trafico['EntryTime'] > fin_periodo) | (Trafico['ExitTime'] < inicio_periodo))]['flightID'].unique()
    # Almacenar las aeronaves pasadas en el diccionario
    aeronaves_por_periodo[inicio_periodo] = aeronaves_pasadas.tolist()

# Crear un DataFrame con los resultados
resultados = pd.DataFrame(aeronaves_por_periodo.items(), columns=['DateTimeFrom', 'Aeronaves'])
# Añadir la columna 'Fin Periodo' al DataFrame con los valores de fin_periodo
resultados.insert(1, 'DateTimeTo', resultados['DateTimeFrom'] + pd.Timedelta(minutes=5)) # 'DateTimeFrom': fecha de inicio del período de 5 min
                                                                                                    # 'DateTimeTo': fecha de fin del período de 5 min
# Eliminar las filas donde la columna 'Aeronaves' tenga una lista vacía
resultados = resultados[resultados['Aeronaves'].apply(len) > 0]


# CREACIÓN DEL DATASET EN EL QUE SE RECOGE, AGRUPADOS EN PERÍODOS DE 5 MIN, LOS VUELOS QUE PASAN POR CADA UNO DE LOS SECTORES

# Crear un nuevo DataFrame desglosando las aeronaves de la lista - una fila por aeronave de la lista
Datos_complejidad = resultados.explode('Aeronaves')

# Ordenar el nuevo DataFrame por la columna 'DateTimeFrom'
Datos_complejidad = Datos_complejidad.sort_values(by='DateTimeFrom')

# Cambiar el nombre de la columna 'Aeronaves' a 'FlightID'
Datos_complejidad = Datos_complejidad.rename(columns={'Aeronaves': 'flightID'})

# Realizar una fusión izquierda basada en la columna 'flightID'
Datos_complejidad = pd.merge(Datos_complejidad, Trafico[['flightID','EntryFL','ExitFL','EntryTrend','ExitTrend']], on='flightID', how='left')

# Realizar una fusión izquierda basada en la columna 'flightID'
Datos_complejidad = pd.merge(Datos_complejidad, Trafico[['flightID', 'assignedFlow', 'assignedFlow_Key']], on='flightID', how='left')

# Separar la columna flightID en dos columnas: flightID y Sector
Datos_complejidad[['flightID', 'Sector']] = Datos_complejidad['flightID'].str.split('_', expand=True)

# Ordenar el DataFrame por las columnas 'Sector' y 'DateTimeFrom'
Datos_complejidad = Datos_complejidad.sort_values(by=['Sector', 'DateTimeFrom'])


# Eliminar duplicados en Trafico basándote en flightID_original
trafico_unique = Trafico.drop_duplicates(subset='flightID_original')
trafico_unique[['DEP', 'ARR']] = trafico_unique['origen_destino'].str.split('-', expand=True) # Separar la columna 'origen_destino' en 'DEP' y 'ARR'
trafico_unique.insert(0, 'flightID_original', trafico_unique.pop('flightID_original')) # Colocar la columna 'flightID_original' en la primera posición
trafico_unique.insert(3, 'DEP', trafico_unique.pop('DEP')) # Colocar la columna 'DEP' en la cuarta posición
trafico_unique.insert(4, 'ARR', trafico_unique.pop('ARR')) # Colocar la columna 'ARR' en la cuarta posición

# Realizar un merge entre Datos_complejidad y Trafico
merged_df = pd.merge(Datos_complejidad, trafico_unique[['flightID_original','origen_destino','DEP','ARR']],left_on='flightID', right_on='flightID_original', how='left')

# Renombrar las columnas para mantener consistencia y eliminar flightID_original si es necesario
merged_df.drop(columns=['flightID_original','origen_destino'], inplace=True)

Datos_complejidad = merged_df




#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- ALGORITMO eCOMMET -------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def calcular_factor_vuelo_evolucion(row, niveles_vuelo_sectores):
    # Obtener los valores de los FL y la tendencia a la entrada y salida del sector
    EntryFL = row['EntryFL']
    ExitFL = row['ExitFL']
    EntryTrend = row['EntryTrend']
    ExitTrend = row['ExitTrend']
    Sector = row['Sector']  # El nombre del sector debe estar en la fila para buscar en el diccionario

    # Calcular la diferencia de FL cruzados
    diferencia_FL = abs(EntryFL - ExitFL)

    # Obtener los límites y la diferencia de FL disponibles en el sector
    limites_sector = niveles_vuelo_sectores.get(Sector, [0, 0, 1])  # Evita divisiones por cero
    niveles_disponibles = limites_sector[2]  # Diferencia de FL en el sector

    # Calcular el porcentaje de FL cruzados
    porcentaje_cruzados = diferencia_FL / niveles_disponibles

    # Si ambos EntryTrend y ExitTrend son CRUISE, asignar complejidad 0
    if EntryTrend == 'CRUISE' and ExitTrend == 'CRUISE':
        return 0

    # Caso de vuelo en evolución (al menos una tendencia es distinta de CRUISE)
    if EntryTrend != 'CRUISE' or ExitTrend != 'CRUISE':
        # Caso especial: Ascenso, crucero y descenso
        if EntryTrend == 'ASCEND' and ExitTrend == 'DESCEND':
            return 0.15
        # Si el número de niveles cruzados es inferior al 50% de los niveles disponibles
        elif porcentaje_cruzados < 0.5:
            return 0.10
        # Si el número de niveles cruzados es igual o superior al 50% de los niveles disponibles
        else:
            return 0.15

    # Si no cumple ninguna de las condiciones anteriores, es decir, la tendencia es CRUISE en ambas, retornar 0
    return 0


def calcular_factor_vuelo_transoceanico(row):
    # Obtener los valores de DEP y ARR
    dep = row['DEP']
    arr = row['ARR']

    # Definir las condiciones
    dep_inicial = dep[:2] if isinstance(dep, str) else ''
    arr_inicial = arr[:2] if isinstance(arr, str) else ''

    if (dep_inicial in ['LE', 'GC'] and arr_inicial[0] in ['T', 'S', 'M', 'K', 'C']) or \
            (arr_inicial in ['LE', 'GC'] and dep_inicial[0] in ['T', 'S', 'M', 'K', 'C']):
        return 0.1
    else:
        return 0


# Convertir el DataFrame de nuevo a un diccionario de diccionarios
matriz_relaciones = matriz_interacciones_flujos.to_dict()

# Crear un DataFrame vacío para almacenar los resultados finales
Complejidad_total = pd.DataFrame()

print(Datos_complejidad['Sector'].unique())
# Iterar sobre cada sector
for sector in Datos_complejidad['Sector'].unique():

    print(f'Sector para el que se calcula la complejidad: '+ sector)
    # Filtrar Datos_complejidad para el sector actual
    Datos_complejidad_sector = Datos_complejidad[Datos_complejidad['Sector'] == sector].copy()

    # Optimización: Filtrar previamente para valores únicos de DateTimeFrom
    unique_datetimefrom = Datos_complejidad_sector['DateTimeFrom'].unique()

    # Crear un DataFrame vacío para almacenar los resultados finales de este sector
    Complejidad_sector = pd.DataFrame()

    for datetimefrom in unique_datetimefrom:
        print(f'Fecha de comienzo del intervalo de cálculo de complejidad: ' + str(datetimefrom))
        # Filtrar Datos_complejidad para el valor de DateTimeFrom actual
        df_temp = Datos_complejidad_sector[Datos_complejidad_sector['DateTimeFrom'] == datetimefrom].copy()

        # Contar cuántas veces aparece cada valor único de assignedFlow
        count_assignedFlow = df_temp['assignedFlow_Key'].value_counts()

        # Crear matriz temporal para esta iteración
        matriz_severidades_ponderadas = pd.DataFrame(0, index=matriz_interacciones_flujos.index,
                                                     columns=matriz_interacciones_flujos.columns)

        for fila in matriz_severidades_ponderadas.index:
            for columna in matriz_severidades_ponderadas.columns:
                flow_count_fila = count_assignedFlow.get(fila, 0)
                flow_count_columna = count_assignedFlow.get(columna, 0)
                if flow_count_fila + flow_count_columna != 0:
                    valor_celda = 0.01 * (
                                matriz_interacciones_flujos.loc[fila, columna] * flow_count_fila * flow_count_columna) / (
                                              flow_count_fila + flow_count_columna)
                else:
                    valor_celda = 0
                matriz_severidades_ponderadas.at[fila, columna] = valor_celda

        matriz_severidades_ponderadas.fillna(0, inplace=True)

        # Crear una lista para almacenar los flujos con interacción para cada vuelo
        flujos_con_interaccion = []

        for index, vuelo in df_temp.iterrows():
            flujo_actual = vuelo['assignedFlow_Key']
            if flujo_actual not in matriz_relaciones:
                flujos_con_interaccion.append([])
                continue

            flujos_interactuantes = [flujo for flujo, interaccion in matriz_relaciones[flujo_actual].items() if
                                     interaccion != 0]
            flujos_con_interaccion.append(flujos_interactuantes)

        df_temp['Flujos_con_interaccion'] = flujos_con_interaccion

        # Crear una lista para almacenar las aeronaves con solape de niveles de vuelo
        aeronaves_con_solape = []

        for index, vuelo in df_temp.iterrows():
            vuelos_con_solape_para_vuelo = []

            for _, otro_vuelo in df_temp.iterrows():
                if (vuelo['EntryFL'] >= vuelo['ExitFL'] and
                        ((otro_vuelo['EntryFL'] < vuelo['ExitFL'] and otro_vuelo['ExitFL'] < vuelo['ExitFL']) or
                         (otro_vuelo['EntryFL'] > vuelo['EntryFL'] and otro_vuelo['ExitFL'] > vuelo['EntryFL'])) or
                        (vuelo['EntryFL'] < vuelo['ExitFL'] and
                         ((otro_vuelo['EntryFL'] < vuelo['EntryFL'] and otro_vuelo['ExitFL'] < vuelo['EntryFL']) or
                          (otro_vuelo['EntryFL'] > vuelo['ExitFL'] and otro_vuelo['ExitFL'] > vuelo['ExitFL'])))):
                    continue
                else:
                    vuelos_con_solape_para_vuelo.append(otro_vuelo['flightID'])

            if vuelo['flightID'] in vuelos_con_solape_para_vuelo:
                vuelos_con_solape_para_vuelo.remove(vuelo['flightID'])

            aeronaves_con_solape.append(vuelos_con_solape_para_vuelo)

        df_temp['AeronavesSolapadas'] = aeronaves_con_solape

        for index, vuelo in df_temp.iterrows():
            flujos_con_interaccion = set(vuelo['Flujos_con_interaccion'])
            aeronaves_solapadas_filtradas = [flightID for flightID in vuelo['AeronavesSolapadas'] if Datos_complejidad_sector.loc[
                Datos_complejidad_sector['flightID'] == flightID, 'assignedFlow_Key'].iloc[0] in flujos_con_interaccion]
            df_temp.at[index, 'AeronavesSolapadas'] = aeronaves_solapadas_filtradas

        num_interacciones_por_flujo = []

        for index, vuelo in df_temp.iterrows():
            interacciones_vuelo = []

            for flujo in matriz_relaciones.keys():
                num_interacciones = len([aeronave for aeronave in vuelo['AeronavesSolapadas'] if
                                         aeronave in df_temp[df_temp['assignedFlow_Key'] == flujo]['flightID'].tolist()])
                interacciones_vuelo.append(num_interacciones)

            num_interacciones_por_flujo.append(interacciones_vuelo)

        df_temp['NumInteraccionesPorFlujo'] = num_interacciones_por_flujo

        producto_escalar = []

        for index, vuelo in df_temp.iterrows():
            flujo_asignado = vuelo['assignedFlow_Key']

            if flujo_asignado not in matriz_severidades_ponderadas:
                producto_escalar.append(0)
                continue

            valores_severidades = [matriz_severidades_ponderadas.at[flujo_asignado, flujo] for flujo in
                                   matriz_severidades_ponderadas.keys()]
            producto = sum(a * b for a, b in zip(vuelo['NumInteraccionesPorFlujo'], valores_severidades))
            producto_escalar.append(producto)

        df_temp['Factor_interaccion'] = producto_escalar

        # Concatenar el df_temp al DataFrame Complejidad
        Complejidad_sector = pd.concat([Complejidad_sector, df_temp], ignore_index=True)

    Complejidad_total = pd.concat([Complejidad_total, Complejidad_sector], ignore_index=True)

    # Cambiar los nan por 0 en los flujos no estándar
    Complejidad_total['assignedFlow_Key'] = Complejidad_total['assignedFlow_Key'].fillna(0)

    # Añadir la columna Factor_complejidad_base con valor 0.8 para todos los elementos
    Complejidad_total['Factor_complejidad_base'] = 0.8

    # Añadir la columna Factor_flujo_no_estándar con valor 0.1 para elementos con 'Sector' en la lista de sectores, y 0 para el resto
    Complejidad_total['Factor_flujo_no_estándar'] = Complejidad_total['Sector'].apply(lambda x: 0 if x in sectores_especificos else 0.1)

    # Aplicar la función al DataFrame para crear la nueva columna del factor de vuelo en evolucion
    Complejidad_total['Factor_vuelo_en_evolución'] = Complejidad_total.apply(calcular_factor_vuelo_evolucion,axis=1,niveles_vuelo_sectores=niveles_vuelo_sectores)

    # Eliminar filas duplicadas basadas en un subconjunto de columnas
    Complejidad_total = Complejidad_total.drop_duplicates(subset=['DateTimeFrom', 'flightID', 'Sector'])

    # Aplicar la función al DataFrame para crear la nueva columna
    Complejidad_total['Factor_vuelo_transoceanico'] = Complejidad_total.apply(calcular_factor_vuelo_transoceanico,axis=1)

    # Añadir la columna Factor_vuelo_militar con valor 0 para todos los elementos, porque no hay vuelo militar
    Complejidad_total['Factor_vuelo_militar'] = 0

    Complejidad_total['Complejidad_total'] = (Complejidad_total['Factor_interaccion'] + Complejidad_total['Factor_complejidad_base'] + Complejidad_total['Factor_flujo_no_estándar'] +
                                              Complejidad_total['Factor_vuelo_en_evolución'] + Complejidad_total['Factor_vuelo_transoceanico'] + Complejidad_total['Factor_vuelo_militar'])

    print(f'Sector ' + sector + ' completado. Se pasa al siguiente sector ....')
    print()

print(f'Cálculo de la complejidad por sector completado. Se guardan los resultados ....')


# Ordenar el DataFrame por las columnas 'Sector' y 'DateTimeFrom'
Complejidad_total = Complejidad_total.sort_values(by=['Sector', 'DateTimeFrom'])

# Agrupar por las columnas 'DateTimeFrom', 'DateTimeTo' y 'Sector' y sumar los valores correspondientes
agrupado = Complejidad_total.groupby(['DateTimeFrom', 'DateTimeTo', 'Sector']).agg({
    'Factor_interaccion': 'sum',
    'Factor_complejidad_base': 'sum',
    'Factor_flujo_no_estándar': 'sum',
    'Factor_vuelo_en_evolución': 'sum',
    'Factor_vuelo_transoceanico': 'sum',
    'Complejidad_total': 'sum'
}).reset_index()

# Renombrar las columnas sumadas para reflejar que son totales
agrupado.rename(columns={
    'Factor_interaccion': 'Factor_interaccion_total',
    'Factor_complejidad_base': 'Factor_complejidad_base_total',
    'Factor_flujo_no_estándar': 'Factor_flujo_no_estándar_total',
    'Factor_vuelo_en_evolución': 'Factor_vuelo_en_evolución_total',
    'Factor_vuelo_transoceanico': 'Factor_vuelo_transoceanico_total',
    'Complejidad_total': 'Complejidad_total'
}, inplace=True)

# El DataFrame resultante es Complejidad_final
Complejidad_final = agrupado

# Ordenar el DataFrame Complejidad_final por DateTimeFrom y por Sector
Complejidad_final.sort_values(by=['Sector', 'DateTimeFrom'], inplace=True)



#%%
##### CALCULO DE COMPLEJIDAD PARA CADA HORA

# Convertir la columna DateTimeFrom a datetime si no lo es
Complejidad_final['DateTimeFrom'] = pd.to_datetime(Complejidad_final['DateTimeFrom'])

# Crear un rango de tiempo para el período de estudio, en intervalos de 1 hora
rango_tiempo_complejidad_hora = pd.date_range(start=start_date, end=end_date, freq='1H')

# Crear un diccionario para almacenar los dataframes dinámicos
Dataframes_complejidad_por_hora = {}

# Iterar sobre cada período de 1 hora
for inicio_periodo in rango_tiempo_complejidad_hora:

    fin_periodo = inicio_periodo + pd.Timedelta(hours=1)

    # Filtrar el dataframe según el período de tiempo
    df_filtrado = Complejidad_final[(Complejidad_final['DateTimeFrom'] >= inicio_periodo) & (Complejidad_final['DateTimeFrom'] <= fin_periodo)]

    # Agrupar por Sector y sumar Complejidad_total
    df_agrupado = df_filtrado.groupby('Sector')['Complejidad_total'].sum().reset_index()

    # Renombrar la columna resultante para mayor claridad
    df_agrupado.rename(columns={'Complejidad_total': 'Suma_Complejidad_total'}, inplace=True)

    Complejidad_por_hora = df_agrupado
    Complejidad_por_hora['Suma_Complejidad_total'] = Complejidad_por_hora['Suma_Complejidad_total'] / 6

    # Crear un nombre dinámico basado en las horas de inicio y fin
    nombre_df = f"Complejidad_{inicio_periodo.hour:02d}_{fin_periodo.hour % 24:02d}"

    # Guardar el dataframe en el diccionario con el nombre dinámico
    Dataframes_complejidad_por_hora[nombre_df] = Complejidad_por_hora


# Guardar cada dataframe del diccionario por separado:
complejidad_por_hora_00_01 = Dataframes_complejidad_por_hora['Complejidad_00_01'].copy()
complejidad_por_hora_01_02 = Dataframes_complejidad_por_hora['Complejidad_01_02'].copy()
complejidad_por_hora_02_03 = Dataframes_complejidad_por_hora['Complejidad_02_03'].copy()
complejidad_por_hora_03_04 = Dataframes_complejidad_por_hora['Complejidad_03_04'].copy()
complejidad_por_hora_04_05 = Dataframes_complejidad_por_hora['Complejidad_04_05'].copy()
complejidad_por_hora_05_06 = Dataframes_complejidad_por_hora['Complejidad_05_06'].copy()
complejidad_por_hora_06_07 = Dataframes_complejidad_por_hora['Complejidad_06_07'].copy()
complejidad_por_hora_07_08 = Dataframes_complejidad_por_hora['Complejidad_07_08'].copy()
complejidad_por_hora_08_09 = Dataframes_complejidad_por_hora['Complejidad_08_09'].copy()
complejidad_por_hora_09_10 = Dataframes_complejidad_por_hora['Complejidad_09_10'].copy()
complejidad_por_hora_10_11 = Dataframes_complejidad_por_hora['Complejidad_10_11'].copy()
complejidad_por_hora_11_12 = Dataframes_complejidad_por_hora['Complejidad_11_12'].copy()
complejidad_por_hora_12_13 = Dataframes_complejidad_por_hora['Complejidad_12_13'].copy()
complejidad_por_hora_13_14 = Dataframes_complejidad_por_hora['Complejidad_13_14'].copy()
complejidad_por_hora_14_15 = Dataframes_complejidad_por_hora['Complejidad_14_15'].copy()
complejidad_por_hora_15_16 = Dataframes_complejidad_por_hora['Complejidad_15_16'].copy()
complejidad_por_hora_16_17 = Dataframes_complejidad_por_hora['Complejidad_16_17'].copy()
complejidad_por_hora_17_18 = Dataframes_complejidad_por_hora['Complejidad_17_18'].copy()
complejidad_por_hora_18_19 = Dataframes_complejidad_por_hora['Complejidad_18_19'].copy()
complejidad_por_hora_19_20 = Dataframes_complejidad_por_hora['Complejidad_19_20'].copy()
complejidad_por_hora_20_21 = Dataframes_complejidad_por_hora['Complejidad_20_21'].copy()
complejidad_por_hora_21_22 = Dataframes_complejidad_por_hora['Complejidad_21_22'].copy()
complejidad_por_hora_22_23 = Dataframes_complejidad_por_hora['Complejidad_22_23'].copy()
complejidad_por_hora_23_00 = Dataframes_complejidad_por_hora['Complejidad_23_00'].copy()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- GUARDAR LAS PREDICCIONES DE COMPLEJIDAD -------------------------------------  #
# -------------------------------------------------------------------------------------------------------------------- #


# FORMATO .pkl
Complejidad_total.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_total_f{fecha_sel}.pkl')
Complejidad_final.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_final_f{fecha_sel}.pkl')
complejidad_por_hora_00_01.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_00-01.pkl')
complejidad_por_hora_01_02.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_01-02.pkl')
complejidad_por_hora_02_03.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_02-03.pkl')
complejidad_por_hora_03_04.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_03-04.pkl')
complejidad_por_hora_04_05.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_04-05.pkl')
complejidad_por_hora_05_06.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_05-06.pkl')
complejidad_por_hora_06_07.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_06-07.pkl')
complejidad_por_hora_07_08.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_07-08.pkl')
complejidad_por_hora_08_09.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_08-09.pkl')
complejidad_por_hora_09_10.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_09-10.pkl')
complejidad_por_hora_10_11.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_10-11.pkl')
complejidad_por_hora_11_12.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_₁₁-₁₂.pkl')
complejidad_por_hora_12_13.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_₁₂-₁₃.pkl')
complejidad_por_hora_13_14.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_₁₃-₁₄.pkl')
complejidad_por_hora_14_15.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_14-15.pkl')
complejidad_por_hora_15_16.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_15-16.pkl')
complejidad_por_hora_16_17.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_16-17.pkl')
complejidad_por_hora_17_18.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_17-18.pkl')
complejidad_por_hora_18_19.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_18-19.pkl')
complejidad_por_hora_19_20.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_19-20.pkl')
complejidad_por_hora_20_21.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_20-21.pkl')
complejidad_por_hora_21_22.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_21-22.pkl')
complejidad_por_hora_22_23.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_22-23.pkl')
complejidad_por_hora_23_00.to_pickle(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_23-00.pkl')


# FORMATO .csv
Complejidad_total.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_total_f{fecha_sel}.csv', index=False, sep=';')
Complejidad_final.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_final_f{fecha_sel}.csv', index=False, sep=';')
complejidad_por_hora_00_01.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_00-01.csv', index=False, sep=';')
complejidad_por_hora_01_02.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_01-02.csv', index=False, sep=';')
complejidad_por_hora_02_03.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_02-03.csv', index=False, sep=';')
complejidad_por_hora_03_04.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_03-04.csv', index=False, sep=';')
complejidad_por_hora_04_05.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_04-05.csv', index=False, sep=';')
complejidad_por_hora_05_06.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_05-06.csv', index=False, sep=';')
complejidad_por_hora_06_07.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_06-07.csv', index=False, sep=';')
complejidad_por_hora_07_08.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_07-08.csv', index=False, sep=';')
complejidad_por_hora_08_09.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_08-09.csv', index=False, sep=';')
complejidad_por_hora_09_10.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_09-10.csv', index=False, sep=';')
complejidad_por_hora_10_11.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_10-11.csv', index=False, sep=';')
complejidad_por_hora_11_12.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_11-12.csv', index=False, sep=';')
complejidad_por_hora_13_14.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_13-14.csv', index=False, sep=';')
complejidad_por_hora_15_16.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_15-16.csv', index=False, sep=';')
complejidad_por_hora_16_17.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_16-17.csv', index=False, sep=';')
complejidad_por_hora_17_18.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_17-18.csv', index=False, sep=';')
complejidad_por_hora_18_19.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_18-19.csv', index=False, sep=';')
complejidad_por_hora_19_20.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_19-20.csv', index=False, sep=';')
complejidad_por_hora_20_21.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_20-21.csv', index=False, sep=';')
complejidad_por_hora_21_22.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_21-22.csv', index=False, sep=';')
complejidad_por_hora_22_23.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_22-23.csv', index=False, sep=';')
complejidad_por_hora_23_00.to_csv(PATH_RESULTADOS_DIA + f'\\Complejidad_por_hora_f{fecha_sel}_23-00.csv', index=False, sep=';')



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- TIEMPO TRANSCURRIDO -----------------------------------------------  #
# -------------------------------------------------------------------------------------------------------------------- #

# Registrar el tiempo de finalización
fin_tiempo = tm.time()
# Calcular la diferencia de tiempo
tiempo_transcurrido = fin_tiempo - inicio_tiempo
print('--------------------------------------------------------------------------------------------')
print('')
print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")