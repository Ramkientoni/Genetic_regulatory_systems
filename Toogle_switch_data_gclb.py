!git clone https://github.com/Ramkientoni/Genetic_regulatory_systems.git

from google.colab import drive
drive.mount('/content/drive')

# Create a folder in the root directory
!mkdir -p "/content/drive/My Drive/Colab Notebooks/Archivos"

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Función para resolver el sistema de ecuaciones
def equations(x, beta, n, K1, K2, gamma1, gamma2, alfa):
    u, v = x
    du_dt = (beta / (1 + (v/K1) ** n)) - gamma1*u
    dv_dt = (alfa / (1 + (u/K2) ** n)) - gamma2*v
    return [du_dt, dv_dt]

# Función para calcular tasas de cambio (du_dt y dv_dt)
def calculate_rates(x, beta, n, K1, K2, gamma1, gamma2, alfa):
    du_dt, dv_dt = equations(x, beta, n, K1, K2, gamma1, gamma2, alfa)
    return du_dt, dv_dt

#Configuracion de parametros
num_iterations = 3000000  # Número de iteraciones para generar datos
chunk_size = 1000  # Tamaño del chunk

#Generación aleatoria de parametros
K1_values = np.random.uniform(0, 1, num_iterations) # Constante de Michaelis-Menten
K2_values = np.random.uniform(0, 1, num_iterations) # Constante de Michaelis-Menten
gamma1_values = np.random.uniform(0, 1, num_iterations) # Tasa de degradación
gamma2_values = np.random.uniform(0, 1, num_iterations) # Tasa de degradación
alfa_values = np.random.uniform(0, 1, num_iterations) # Tasa de sintesis
beta_values = np.random.uniform(0, 1, num_iterations) # Tasa de sintesis
n_values = [2, 3, 4] # Coeficiente de Hill

#Eliminar duplicados manteniendo el orden
K1_values = np.unique(K1_values)
K2_values = np.unique(K2_values)
gamma1_values = np.unique(gamma1_values)
gamma2_values = np.unique(gamma2_values)
alfa_values = np.unique(alfa_values)
beta_values = np.unique(beta_values)

while len(K1_values) < num_iterations:
    additional_samples = num_iterations - len(K1_values)
    new_K1_values = np.random.uniform(0, 1, additional_samples)
    K1_values = np.concatenate((K1_values, new_K1_values))

while len(K2_values) < num_iterations:
    additional_samples = num_iterations - len(K2_values)
    new_K2_values = np.random.uniform(0, 1, additional_samples)
    K2_values = np.concatenate((K2_values, new_K2_values))

while len(gamma1_values) < num_iterations:
    additional_samples = num_iterations - len(gamma1_values)
    new_gamma1_values = np.random.uniform(0, 1, additional_samples)
    gamma1_values = np.concatenate((gamma1_values, new_gamma1_values))

while len(gamma2_values) < num_iterations:
    additional_samples = num_iterations - len(gamma2_values)
    new_gamma2_values = np.random.uniform(0, 1, additional_samples)
    gamma2_values = np.concatenate((gamma2_values, new_gamma2_values)) 

while len(alfa_values) < num_iterations:
    additional_samples = num_iterations - len(alfa_values)
    new_alfa_values = np.random.uniform(0, 1, additional_samples)
    alfa_values = np.concatenate((alfa_values, new_alfa_values))       

while len(beta_values) < num_iterations:
    additional_samples = num_iterations - len(beta_values)
    new_beta_values = np.random.uniform(0, 1, additional_samples)
    beta_values = np.concatenate((beta_values, new_beta_values)) 

K1_values = K1_values[:num_iterations]
gamma_values = gamma_values[:num_iterations]
alfa_values = alfa_values[:num_iterations]
beta_values = beta_values[:num_iterations]

#Generación y división de datos
# Ruta de la carpeta en Google Drive donde deseas guardar el archivo CSV
google_drive_folder = '/content/drive/MyDrive/Colab Notebooks/Archivos/'

# Crear archivo CSV para almacenar datos
csv_filename = os.path.join(google_drive_folder, 'datos_puntos_estacionarios_verificados.csv')

# Comprobar si el archivo CSV ya existe
if not os.path.isfile(csv_filename):
    # Si no existe, crea el encabezado
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['u', 'v', 'du_dt', 'dv_dt', 'n', 'beta', 'alfa', 'K1', 'K2', 'gamma1', 'gamma2', 'split'])  # Agregar columna 'split' y agregar las derivadas

# Inicializar lista para almacenar datos
data_buffer = np.array([]).reshape(0,10)

for n in n_values:

    # Generar y dividir datos en chunks
    for i in range(num_iterations):
        K1 = K1_values[i]
        K2 = K2_values[i]
        gamma1 = gamma1_values[i]
        gamma2 = gamma2_values[i]
        alfa = alfa_values[i]
        beta = beta_values[i]        

        # Resolver el sistema de ecuaciones para obtener u y v
        initial_guess = [0.0, 0.0]
        solution = fsolve(equations, initial_guess, args=(beta, n, K1, K2, gamma1, gamma2, alfa))
        u, v = solution

        # Calcular las derivadas
        du_dt, dv_dt = calculate_rates(solution, beta, n, K1, K2, gamma1, gamma2, alfa)

        # Agregar datos al búfer con el identificador de conjunto
        data_entry = np.array([u, v, du_dt, dv_dt, n, beta, K1, K2, gamma1, gamma2, alfa, 0])  # 0 para conjunto de entrenamiento
        data_buffer = np.vstack((data_buffer, data_entry))

        # Si se ha acumulado suficiente cantidad de datos en el búfer, escribirlos en el archivo CSV y vaciar el búfer
        if data_buffer.shape[0] >= chunk_size:

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
            data_buffer = np.array([]).reshape(0,10) # Vaciar Lista

    # Si hay datos restantes en el búfer, escribirlos en el archivo CSV
    if data_buffer.shape[0] > 0:
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
filtered_df = df[df['K1'] + df['K2'] > df['u'] + df['v']]

# Contar la cantidad de datos por etiqueta
counts = df['split'].value_counts()
counts_filter = filtered_df['split'].value_counts()

# Imprimir los resultados
print("Cantidad de datos generados para entrenamiento:", counts[0])
print("Cantidad de datos generados para validacion:", counts[1])
print("Cantidad de datos generados para prueba:", counts[2])
print("Generación y escritura de datos completadas.")

print("Cantidad de datos verificados para entrenamiento:", counts_filter[0])
print("Cantidad de datos verificados para validacion:", counts_filter[1])
print("Cantidad de datos verificados para prueba:", counts_filter[2])
print("Generación y escritura de datos completadas.")

# Crear gráficos del espacio fase
plt.figure(figsize=(12, 4))

# Gráfico 1: u vs du_dt
plt.subplot(131)
plt.scatter(df['u'], df['du_dt'], c=df['split'], cmap='viridis')
plt.xlabel('u')
plt.ylabel('du_dt')
plt.title('Espacio Fase: u vs du_dt')

# Gráfico 2: v vs dv_dt
plt.subplot(132)
plt.scatter(df['v'], df['dv_dt'], c=df['split'], cmap='viridis')
plt.xlabel('v')
plt.ylabel('dv_dt')
plt.title('Espacio Fase: v vs dv_dt')

# Gráfico 3: du_dt vs dv_dt
plt.subplot(133)
plt.scatter(df['du_dt'], df['dv_dt'], c=df['split'], cmap='viridis')
plt.xlabel('du_dt')
plt.ylabel('dv_dt')
plt.title('Espacio Fase: du_dt vs dv_dt')

plt.tight_layout()
plt.show()

Extraer las columnas u, v, du_dt y dv_dt
u = df['u']
v = df['v']
du_dt = df['du_dt']  # Asegúrate de tener la columna 'du_dt' en tu archivo CSV
dv_dt = df['dv_dt']  # Asegúrate de tener la columna 'dv_dt' en tu archivo CSV

# Crear un gráfico de vectores (campo vectorial)
plt.figure(figsize=(8, 6))
plt.quiver(u, v, du_dt, dv_dt, scale=20, angles='xy', scale_units='xy')

# Etiquetas y título
plt.xlabel('u')
plt.ylabel('v')
plt.title('Espacio Fase con Campo Vectorial')

# Mostrar el gráfico
plt.grid(True)
plt.show()
          
