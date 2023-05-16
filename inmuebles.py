import pandas as pd
properties = pd.read_csv('inmuebles_bogota.csv')
print(properties.head())
print('row, Columns dataset: \n',properties.shape)
print('Columns dataset: \n',properties.columns)
columnas = {'Baños':'Banos', 'Área':'Area'}
properties = properties.rename(columns=columnas)
print(properties.sample(10))
print('Information dataset: ')
print(properties.info())

n = 123
m= 126
# print(f'Information property number {n} : , properties.iloc[n])
print(f'Information from property {n} until property {m}: \n', properties.iloc[n:m])

# Valuue of property number n
print(f'\n Value of property number {n} : ', properties['Valor'][n])

#mean of area of properties
print(f'\n mean of area of all properties :', properties.Area.mean() ,'\n')

print(properties.sample(100))

print('\nNumber of properties when barrio = chico reservado: ' , sum(properties.Barrio == 'Chico Reservado'))

inmuebles_chico = (properties.Barrio == 'Chico Reservado')

chico_reservado = properties[inmuebles_chico]
print(chico_reservado)

print('\nArea media en chico reservado: ', chico_reservado.Area.mean())

print('\nArea media general', properties.Area.mean())

print('\nnumero de barrios totales: ', len(properties.Barrio.value_counts()))


inmuebles_barrio = properties.Barrio.value_counts()
print(inmuebles_barrio)

inmuebles_barrio.plot.bar()

inmuebles_barrio.head(10).plot.bar()

valor = properties.Valor.str.split(expand=True)
properties['Moneda'] = valor[0]
properties['Precio'] = valor[1]
print(properties.sample(3))

#string to float, value in millions
properties['Precio'] = properties['Precio'].str.replace('.','',regex=True)
print(properties[['Precio','Barrio']])
properties['Precio_Millon'] = properties.Precio.astype('float')/1000000
print(properties.info())

#descriptive stadistics
print(properties.describe())

#descriptive stadistics with 2 decimals
pd.set_option('display.precision',2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(properties.describe())


properties.loc[properties.Habitaciones == 110]
properties.loc[properties.Area == 2]
#price Histogram
properties['Precio_Millon'].plot.hist(bins=10)

#Distribución de Valores de los properties en Bogotá
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
grafica = sns.histplot(data=properties, x='Precio_Millon', kde=True, hue='Tipo')
grafica.set_title('Distribución de Valores de los properties en Bogotá')
plt.xlim((50,1000))
#plt.savefig('C:\Users\ander\Análisis de datos/valor_inmuebles.png', format='png')
plt.show()

#valor metro cuadrado
properties['Valor_m2_Millon'] = properties['Precio_Millon']/properties['Area'] 
properties.head(3)
#promedio de valores para cada barrio
properties.groupby('Barrio').mean()
#suma de valores para cada barrio
datos_barrio = properties.groupby('Barrio').sum()
print(datos_barrio)
#valor del metro cuadrado por barrio
datos_barrio['Valor_m2_Barrio'] = datos_barrio['Precio_Millon']/datos_barrio['Area']
print(datos_barrio)
print('datos_barrio')
m2_barrio = dict(datos_barrio['Valor_m2_Barrio'])
#Agrega el valor por metro cuadrado en cada barrio
properties['Valor_m2_Barrio'] = properties['Barrio']
properties['Valor_m2_Barrio'] = properties['Valor_m2_Barrio'].map(m2_barrio)
properties.head(5)
print()
#los 10 barrios que mas ofertan apartamentos
top_barrios = properties['Barrio'].value_counts()[:10].index
print(top_barrios)

datos_barrio.reset_index(inplace=True)
print(datos_barrio)
#consulta los datos del top de barrios
print(datos_barrio.query('Barrio in @top_barrios'))

#grafica el valor del metro cuadrado por barrio
plt.figure(figsize=(10,8))
ax = sns.barplot(x="Barrio", y="Valor_m2_Barrio", data = datos_barrio.query('Barrio in @top_barrios'))
ax.tick_params(axis='x', rotation=45)

#variacion y mediana
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="Barrio", y="Valor_m2_Millon", data = properties.query('Barrio in @top_barrios'))
ax.tick_params(axis='x', rotation=45)
plt.show()

#variacion y mediana cuando valor por metro cuadrado menor que 15 millones
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="Barrio", y="Valor_m2_Millon", data = properties.query('Barrio in @top_barrios & Valor_m2_Millon < 15'))
ax.tick_params(axis='x', rotation=45)
plt.show()

#variacion y mediana cuando area es menor que 500 metros
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="Barrio", y="Area", data = properties.query('Barrio in @top_barrios & Area < 500'))
ax.tick_params(axis='x', rotation=45)
plt.show()

#variacion y mediana cuando precio en millones es menor a 2000
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="Barrio", y="Precio_Millon", data = properties.query('Barrio in @top_barrios & Precio_Millon < 2000'))
ax.tick_params(axis='x', rotation=45)
plt.show()

datos_raw = pd.read_csv('Ident (Cap A).csv', sep = ';',encoding='latin-1')
print(datos_raw.head())

#carga datos del dane colombia
datos_b = pd.read_csv('Datos de la vivenda y su entorno (Capitulo B).csv',sep=';',encoding='latin-1')
datos_c = pd.read_csv('Condiciones habitacionales del hogar (Capitulo C).csv',sep=';',encoding='latin-1')
datos_e = pd.read_csv('Composicion del hogar y demografia (Capitulo E).csv',sep=';',encoding='latin-1')
datos_h = pd.read_csv('Educacion (Capitulo H).csv',sep=';',encoding='latin-1')
datos_l = pd.read_csv('Percepcion sobre las condiciones de vida y el desempeno institucional (Capitulo L).csv',sep=';',encoding='latin-1')
datos_k = pd.read_csv('Fuerza de trabajo (Capitulo K).csv',sep=';',encoding='latin-1')

datos_dane = pd.merge(datos_raw,datos_b,on='DIRECTORIO', how='left')
datos_dane = pd.merge(datos_dane,datos_c,on='DIRECTORIO', how='left')
datos_dane = pd.merge(datos_dane,datos_e,on='DIRECTORIO', how='left')

datos_dane = pd.read_csv('datos_dane.csv')
print(datos_dane.info())

#cambia el nombre de las columnas
dic_dane = {
       'NVCBP4':'CONJUNTO_CERRADO',
       'NVCBP14A':'FABRICAS_CERCA', 'NVCBP14D':'TERMINALES_BUS', 'NVCBP14E':'BARES_DISCO', 
       'NVCBP14G':'OSCURO_PELIGROSO', 'NVCBP15A':'RUIDO', 'NVCBP15C':'INSEGURIDAD',
       'NVCBP15F':'BASURA_INADECUADA', 'NVCBP15G':'INVASION','NVCBP16A3':'MOV_ADULTOS_MAYORES', 
       'NVCBP16A4':'MOV_NINOS_BEBES',
       'NPCKP17':'OCUPACION','NPCKP18':'CONTRATO','NPCKP23':'SALARIO_MES', 
       'NPCKP44A':'DONDE_TRABAJA', 'NPCKPN62A':'DECLARACION_RENTA', 
       'NPCKPN62B':'VALOR_DECLARACION', 'NPCKP64A':'PERDIDA_TRABAJO_C19', 
       'NPCKP64E':'PERDIDA_INGRESOS_C19',
       'NHCCP3':'TIENE_ESCRITURA', 'NHCCP6':'ANO_COMPRA', 'NHCCP7':'VALOR_COMPRA', 'NHCCP8_1':'HIPOTECA_CRED_BANCO',
       'NHCCP8_2':'OTRO_CRED_BANCO', 'NHCCP8_3':'CRED_FNA', 'NHCCP8_6':'PRESTAMOS_AMIGOS',
       'NHCCP8_7':'CESANTIAS', 'NHCCP8_8':'AHORROS', 'NHCCP8_9':'SUBSIDIOS',
       'NHCCP9':'CUANTO_PAGARIA_MENSUAL', 'NHCCP11':'PLANES_ADQUIRIR_VIVIENDA', 
       'NHCCP11A':'MOTIVO_COMPRA', 'NHCCP12':'RAZON_NO_ADQ_VIV', 'NHCCP41':'TIENE_CARRO','NHCCP41A':'CUANTOS_CARROS',
       'NHCCP47A':'TIENE_PERROS', 'NHCCP47B':'TIENE_GATOS', 'NHCLP2A':'VICTIMA_ATRACO', 'NHCLP2B':'VICTIMA_HOMICIDIO', 
       'NHCLP2C':'VICTIMA_PERSECUSION',
       'NHCLP2E':'VICTIMA_ACOSO', 'NHCLP4':'COMO_VIVE_ECON', 'NHCLP5':'COMO_NIVEL_VIDA', 
       'NHCLP8AB':'REACCION_OPORTUNA_POLICIA', 'NHCLP8AE':'COMO_TRANSPORTE_URBANO', 'NHCLP10':'SON_INGRESOS_SUFICIENTES',
       'NHCLP11':'SE_CONSIDERA_POBRE', 'NHCLP29_1A':'MED_C19_TRABAJO', 
       'NHCLP29_1C':'MED_C19_CAMBIO_VIVIENDA', 'NHCLP29_1E':'MED_C19_ENDEUDAMIENTO', 
       'NHCLP29_1F':'MED_C19_VENTA_BIENES','NPCHP4':'NIVEL_EDUCATIVO'
       }

datos_dane = datos_dane.rename(columns=dic_dane)
print(datos_dane.columns)
print(datos_dane.info())
datos_dane.groupby('NOMBRE_ESTRATO')[['CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO','OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19','PLANES_ADQUIRIR_VIVIENDA']].mean().head()

#reemplazar por cero los valores 2
datos = datos_dane[['NOMBRE_ESTRATO','CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO','OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19','PLANES_ADQUIRIR_VIVIENDA']].replace(2,0)
print(datos)
print(datos.loc[datos.NOMBRE_ESTRATO == '20 de Julio'])

datos_tratados = datos.groupby('NOMBRE_ESTRATO')[['CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO','OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19','PLANES_ADQUIRIR_VIVIENDA']].mean()
print(datos_tratados)

#une los datos del dane con los datos iniciales
datos_ml = pd.merge(properties,datos_tratados, left_on='UPZ', right_on='NOMBRE_ESTRATO', how='left')
print(datos_ml)

#agrega código upz
upz = pd.read_csv('cod_upz.csv')
datos_ml = pd.merge(datos_ml,upz,left_on='UPZ',right_on='NOMBRE_ESTRATO', how='inner')
print(datos_ml.head())
print(datos_ml.info())

plt.figure(figsize=(10,8))
sns.boxplot(data=datos_ml, y = 'Precio_Millon')
plt.show()

#consulta precios entre 60 y 5000 millones
print(datos_ml.query('Precio_Millon > 5000 | Precio_Millon < 60'))

datos_ml = datos_ml.query('Precio_Millon < 1200 & Precio_Millon > 60')
print(datos_ml)

plt.figure(figsize=(10,8))
sns.boxplot(data=datos_ml, y = 'Precio_Millon')
plt.show()
#salario anual en millones
datos_ml['SALARIO_ANUAL_MI'] = datos_ml['SALARIO_MES']*12/1000000
print(datos_ml['SALARIO_ANUAL_MI'])
plt.figure(figsize=(10,8))
sns.scatterplot(data=datos_ml, x='SALARIO_ANUAL_MI',y ='Valor_m2_Millon')
plt.ylim((0,15))
plt.show()

#correlacíon de los datos
print(datos_ml.corr())
