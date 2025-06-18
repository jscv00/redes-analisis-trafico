import pandas
import numpy

"""
5000 registros para el analisis de trafico
"""

numpy.random.seed(42)

"""
Generacion de datos
"""
hora = numpy.random.randint(8, 18, size=5000)
dispositivos = numpy.random.randint(100, 1500, size=5000)
ancho_banda = numpy.random.randint(10, 100, size=5000)
tipo_trafico = numpy.random.choice(
    ['web', 'video', 'audio', 'ssh', 'ftp'], size=5000)

"""
Definir el trafico en proporciones 
"""
coef_tipo_trafico = {'web': 100000, 'video': 200000,
                     'audio': 150000, 'ssh': 250000, 'ftp': 300000}
ruido = numpy.random.normal(0, 50000, size=5000)

"""
Generacion del trafico con cada una de las variables
"""
trafico_datos = (dispositivos * 200) + (ancho_banda * 500)+(hora * 5000) + \
    numpy.array([coef_tipo_trafico[tipo] for tipo in tipo_trafico]) + ruido

"""
Creacion del DataFrame
"""
data = pandas.DataFrame({
    'Hora del día': hora,
    'Dispositivos': dispositivos,
    'Ancho de banda (Mbps)': ancho_banda,
    'Tipo de tráfico': tipo_trafico,
    'Tráfico de datos (Bytes)': trafico_datos
})

"""
Guardar datos en un archivo CSV
"""
data.to_csv('trafico_datos_regresion_modificado.csv', index=False)

"""
Comprobar los datos generados
"""
print(data.head())
