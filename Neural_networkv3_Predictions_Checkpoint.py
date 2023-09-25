import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# Cargar datos generados previamente desde el archivo CSV
csv_filename = 'datos_puntos_estacionarios.csv'
df = pd.read_csv(csv_filename)

# Separar características (parámetros) y etiquetas (u, v, split)
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

# Definir el modelo
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)  # Dos salidas, u y v
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Definir el callback para guardar el modelo
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.h5',  # Ruta donde se guarda el modelo
    monitor='val_loss',  # Métrica para controlar (puedes cambiarla a otra métrica si es necesario)
    save_best_only=True,  # Guardar solo el mejor modelo basado en la métrica monitoreada
    save_weights_only=False,  # Guardar el modelo completo en lugar de solo los pesos
    verbose=1  # Muestra mensajes durante el proceso
)

# Entrenar el modelo con el callback
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback]  # Agregar el callback aquí
)

# Guardar la historia de entrenamiento en un archivo CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('history.csv', index=False)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")

# Usar el modelo para hacer predicciones
predictions = model.predict(X_test)

# Guardar la configuración del modelo en un archivo JSON
model_config = {
    'layers': [layer.get_config() for layer in model.layers],
    'optimizer': model.optimizer.get_config(),
    'loss': model.loss
}
with open('model_config.json', 'w') as config_file:
    json.dump(model_config, config_file)

# Registrar métricas de evaluación en un archivo
with open('evaluation_metrics.txt', 'w') as metrics_file:
    metrics_file.write(f'Pérdida en el conjunto de prueba: {loss}\n')

# Visualizar las predicciones
# Indexar las predicciones y los valores reales para 'u' y 'v'
predictions_u = predictions[:, 0]
predictions_v = predictions[:, 1]
y_test_u = y_test[:, 0]
y_test_v = y_test[:, 1]

# Crear gráficos de dispersión para 'u'
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_u, predictions_u, alpha=0.5)
plt.xlabel('Valores reales (u)')
plt.ylabel('Predicciones (u)')
plt.title('Predicciones vs. Valores reales para u')

# Crear gráficos de dispersión para 'v'
plt.subplot(1, 2, 2)
plt.scatter(y_test_v, predictions_v, alpha=0.5)
plt.xlabel('Valores reales (v)')
plt.ylabel('Predicciones (v)')
plt.title('Predicciones vs. Valores reales para v')

plt.tight_layout()
plt.show()
