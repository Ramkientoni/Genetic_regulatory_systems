import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Cargar datos generados previamente desde el archivo CSV
csv_filename = 'datos_puntos_estacionarios.csv'
df = pd.read_csv(csv_filename)

# Separar características (parámetros) y etiquetas (u, v)
X = df[['K', 'beta', 'gamma', 'alfa', 'n', 'split']].values
y = df[['u', 'v', 'split']].values

# Definir los datos de entrenamiento, validación y prueba
# Seleccionar filas donde split es 0 y quitar la última columna (split)
X_train = X[X[:, -1] == 0][:, :-1]  
y_train = y[y[:, -1] == 0][:, :-1]

# Seleccionar filas donde split es 1 y quitar la última columna (split)
X_val = X[X[:, -1] == 1][:, :-1]  
y_val = y[y[:, -1] == 1][:, :-1]

# Seleccionar filas donde split es 2 y quitar la última columna (split)
X_test = X[X[:, -1] == 2][:, :-1]  
y_test = y[y[:, -1] == 2][:, :-1]


# Definir la arquitectura de la red neuronal
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)  # Dos salidas, u y v
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")

# Usar el modelo para hacer predicciones
predictions = model.predict(X_test)

# Aquí puedes realizar cualquier otra tarea de postprocesamiento o análisis de resultados según tus necesidades.
