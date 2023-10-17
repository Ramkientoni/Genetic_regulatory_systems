import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import csv
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Función para resolver el sistema de ecuaciones
def equations(x, beta, n, K, gamma, alfa):
    u, v = x
    du_dt = (beta / (1 + (v/K) ** n)) - gamma*u
    dv_dt = (alfa / (1 + (u/K) ** n)) - gamma*v
    return [du_dt, dv_dt]

# Parámetros
num_iterations = 5000000  # Número de iteraciones para generar datos
chunk_size = 1000  # Tamaño del chunk
n = 2

# Generar valores aleatorios para los parámetros
K_values = np.random.uniform(0, 1, num_iterations)
beta_values = np.random.uniform(0, 1, num_iterations)
gamma_values = np.random.uniform(0, 1, num_iterations)
alfa_values = np.random.uniform(0, 1, num_iterations)

# Eliminar duplicados manteniendo el orden de aparición
K_values = np.unique(K_values)
beta_values = np.unique(beta_values)
gamma_values = np.unique(gamma_values)
alfa_values = np.unique(alfa_values)

# Asegurar que tengas al menos num_samples valores únicos
while len(K_values) < num_iterations:
    additional_samples = num_iterations - len(K_values)
    new_K_values = np.random.uniform(0, 1, additional_samples)
    K_values = np.concatenate((K_values, new_K_values))

# Asegurar que todas las listas de valores tengan la misma longitud
K_values = K_values[:num_iterations]
beta_values = beta_values[:num_iterations]
gamma_values = gamma_values[:num_iterations]
alfa_values = alfa_values[:num_iterations]

# Crear archivo CSV para almacenar datos
csv_filename = 'datos_puntos_estacionarios_5mill.csv'

# Comprobar si el archivo CSV ya existe
if not os.path.isfile(csv_filename):
    # Si no existe, crea el encabezado
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['u', 'v', 'du_dt', 'dv_dt' 'n', 'beta', 'K', 'gamma', 'alfa', 'split'])  # Agregar columna 'split'

# Inicializar lista para almacenar datos
data_buffer = []

# Generar, normalizar y dividir datos en chunks
for i in range(num_iterations):
    K = K_values[i]
    beta = beta_values[i]
    gamma = gamma_values[i]
    alfa = alfa_values[i]

    # Resolver el sistema de ecuaciones para obtener u y v
    initial_guess = [0.0, 0.0]
    u, v = fsolve(equations, initial_guess, args=(beta, n, K, gamma, alfa))

    # Agregar datos al búfer con el identificador de conjunto
    data_buffer.append([u, v, du_dt, dv_dt, n, beta, K, gamma, alfa, 0])  # 0 para conjunto de entrenamiento

    # Si se ha acumulado suficiente cantidad de datos en el búfer, escribirlos en el archivo CSV y vaciar el búfer
    if len(data_buffer) >= chunk_size:

        # Dividir datos en conjuntos de entrenamiento, validación y prueba
        X_train, X_temp, y_train, y_temp = train_test_split(data_buffer[:, :-1], data_buffer[:, :2], test_size=0.1, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Asignar identificador de conjunto: 0 para entrenamiento, 1 para validación, 2 para prueba
        for j in range(len(X_train)):
            data_buffer[j][-1] = 0

        for j in range(len(X_train), len(X_train) + len(X_val)):
            data_buffer[j][-1] = 1

        for j in range(len(X_train) + len(X_val), len(data_buffer)):
            data_buffer[j][-1] = 2

        # Guardar datos con identificador de conjunto en el archivo CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_buffer)
        data_buffer = []

# Si hay datos restantes en el búfer, escribirlos en el archivo CSV
if data_buffer:

    # Dividir datos en conjuntos de entrenamiento, validación y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(data_buffer[:, :-1], data_buffer[:, :2], test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Asignar identificador de conjunto: 0 para entrenamiento, 1 para validación, 2 para prueba
    for j in range(len(X_train)):
        data_buffer[j][-1] = 0

    for j in range(len(X_train), len(X_train) + len(X_val)):
        data_buffer[j][-1] = 1

    for j in range(len(X_train) + len(X_val), len(data_buffer)):
        data_buffer[j][-1] = 2

    # Guardar datos con identificador de conjunto en el archivo CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_buffer)

df = pd.read_csv(csv_filename)

# Contar la cantidad de datos por etiqueta
counts = df['split'].value_counts()

# Imprimir los resultados
print("Cantidad de datos para entrenamiento:", counts[0])
print("Cantidad de datos para validacion:", counts[1])
print("Cantidad de datos para prueba:", counts[2])
print("Generación y escritura de datos completadas.")

# Graficar espacio fase
# Extraer las columnas relevantes
u = df['u']
v = df['v']
du_dt = df['du_dt']  # Asegúrate de tener la columna 'du_dt' en tu archivo CSV
dv_dt = df['dv_dt']  # Asegúrate de tener la columna 'dv_dt' en tu archivo CSV

# Crear gráficos de dispersión
plt.figure(figsize=(15, 5))

# Gráfico 1: u vs du_dt
plt.subplot(131)
plt.scatter(u, du_dt, alpha=0.5)
plt.xlabel('u')
plt.ylabel('du_dt')
plt.title('u vs du_dt')

# Gráfico 2: v vs dv_dt
plt.subplot(132)
plt.scatter(v, dv_dt, alpha=0.5)
plt.xlabel('v')
plt.ylabel('dv_dt')
plt.title('v vs dv_dt')

# Gráfico 3: du_dt vs dv_dt
plt.subplot(133)
plt.scatter(du_dt, dv_dt, alpha=0.5)
plt.xlabel('du_dt')
plt.ylabel('dv_dt')
plt.title('du_dt vs dv_dt')

plt.tight_layout()
plt.show()
