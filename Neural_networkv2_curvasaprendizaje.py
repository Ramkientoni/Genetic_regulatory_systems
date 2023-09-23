import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Cargar datos generados previamente desde el archivo CSV
csv_filename = 'datos_puntos_estacionarios.csv'
df = pd.read_csv(csv_filename)

# Separar características (parámetros) y etiquetas (u, v, split)
X = df[['K', 'beta', 'gamma', 'alfa', 'n', 'split']].values
y = df[['u', 'v', 'split']].values

# Definir los datos de entrenamiento, validación y prueba
X_train = X[X[:, -1] == 0][:, :-1]  
y_train = y[y[:, -1] == 0][:, :-1]
X_val = X[X[:, -1] == 1][:, :-1]  
y_val = y[y[:, -1] == 1][:, :-1]
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

# Listas para almacenar la pérdida en cada época
losses = []

# Tamaños de conjunto de entrenamiento que deseas probar
train_sizes = [100, 200, 500, 1000, 2000, 5000]

for train_size in train_sizes:
    # Seleccionar un subconjunto de datos de entrenamiento
    X_train_subset = X_train[:train_size]
    y_train_subset = y_train[:train_size]

    # Entrenar el modelo con el subconjunto de datos
    history = model.fit(X_train_subset, y_train_subset, epochs=50, validation_data=(X_val, y_val), verbose=0)

    # Registra la pérdida en cada época
    losses.append(history.history['loss'])

# Dibuja la curva de aprendizaje
plt.figure(figsize=(8, 6))

for i, train_size in enumerate(train_sizes):
    plt.plot(losses[i], label=f'Train Size: {train_size}')

plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()