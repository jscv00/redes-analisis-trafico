import pandas
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

"""
Cargar los datos que se generaron en el archivo CSV
"""
data = pandas.read_csv('trafico_datos_regresion_modificado.csv')

"""
Determinar y convertir las variables categoricas a numericas

    'Hora del día': hora,
    'Dispositivos': dispositivos,
    'Ancho de banda (Mbps)': ancho_banda,
    'Tipo de tráfico': tipo_trafico,
    'Tráfico de datos (Bytes)': trafico_datos
"""
data['Tipo de tráfico'] = data['Tipo de tráfico'].map(
    {'web': 0, 'video': 1, 'audio': 2, 'ssh': 3, 'ftp': 4})

"""
Definir las variables independientes (X) y dependientes (y)
"""
X = data[['Hora del día', 'Dispositivos',
          'Ancho de banda (Mbps)', 'Tipo de tráfico']]
y = data['Tráfico de datos (Bytes)']


"""
Dividir los datos en conjuntos de entrenamiento y prueba
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


"""
Crear el modelo de regresión lineal
"""
model = LinearRegression()

"""
Se entrena el modelo con los datos de entrenamiento
"""
model.fit(X_train, y_train)


"""
Realizar predicciones con el conjunto de prueba
"""
y_pred = model.predict(X_test)


"""
Evaluar el modelo
"""
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"""
Los coeficientes del modelo son: {model.coef_}.

Intersección del modelo: {model.intercept_}.

El error cuadrático medio (MSE) es: {mse}.

El coeficiente de determinación (R^2) es: {r2}.
""")


"""
Visiualizar los resultados a través de un gráfico de dispersión
"""

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Datos de prueba')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red',
         linestyle='--', label='Línea de referencia')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Regresión Lineal: Valores Reales vs Predicciones')
plt.legend()
plt.grid()
plt.show()

# Configuración del gráfico
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, color='#6B7280', alpha=0.6,
#             label='Datos de prueba')  # Verde oliva suave
# plt.plot([y.min(), y.max()], [y.min(), y.max()], color='#D97706',
#          linestyle='--', label='Línea de referencia')  # Dorado cálido
# plt.xlabel('Valores reales', fontsize=12, color='#333333')
# plt.ylabel('Predicciones', fontsize=12, color='#333333')
# plt.title('Regresión Lineal: Valores Reales vs Predicciones',
#           fontsize=14, color='#333333')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7,
#          color='#D1D5DB')  # Rejilla en gris claro
# plt.style.use('ggplot')  # Estilo ggplot para un look limpio
# plt.show()
