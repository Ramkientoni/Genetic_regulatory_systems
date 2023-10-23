!git clone https://github.com/Ramkientoni/Genetic_regulatory_systems.git

from google.colab import drive
drive.mount('/content/drive')

# Create a folder in the root directory
!mkdir -p "/content/drive/My Drive/Colab Notebooks/Archivos"

import csv
import os
import numpy as np
import pandas as pd
from math import sqrt
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

# Función para calcular eigenvalores
def calculate_eigenvalues(x, beta, n, K1, K2, gamma1, gamma2, alfa):
    # Obtén las tasas de cambio du_dt y dv_dt en los puntos dados por la solución
    du_dt, dv_dt = calculate_rates(x, beta, n, K1, K2, gamma1, gamma2, alfa)
    J = np.array([[du_dt, -x[0] * beta * n * x[1] ** n / (K1 ** n * (1 + (x[1] / K1) ** n) ** 2)], [x[1] * alfa * n * x[0] ** n / (K2 ** n * (1 + (x[0] / K2) ** n) ** 2), dv_dt]])
    eigenvalues = np.linalg.eigvals(J)
    return eigenvalues

num_iterations = 5000 # Número de iteraciones para generar datos
chunk_size = 1000  # Tamaño del chunk

K1_values = np.random.uniform(0, 1, num_iterations) # Constante de Michaelis-Menten
K2_values = np.random.uniform(0, 1, num_iterations) # Constante de Michaelis-Menten
gamma1_values = np.random.uniform(0, 1, num_iterations) # Tasa de degradación
gamma2_values = np.random.uniform(0, 1, num_iterations) # Tasa de degradación
alfa_values = np.random.uniform(0, 1, num_iterations) # Tasa de sintesis
beta_values = np.random.uniform(0, 1, num_iterations) # Tasa de sintesis
n_values = [2, 3, 4] # Coeficiente de Hill

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
K2_values = K2_values[:num_iterations]
gamma1_values = gamma1_values[:num_iterations]
gamma2_values = gamma2_values[:num_iterations]
alfa_values = alfa_values[:num_iterations]
beta_values = beta_values[:num_iterations]

# Ruta de la carpeta en Google Drive donde deseas guardar el archivo CSV
google_drive_folder = '/content/drive/MyDrive/Colab Notebooks/Archivos/'

# Crear archivo CSV para almacenar datos
csv_filename = os.path.join(google_drive_folder, 'soluciones_Toggle_switch.csv')

# Comprobar si el archivo CSV ya existe
if not os.path.isfile(csv_filename):
    # Si no existe, crea el encabezado
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['u', 'v', 'stability identifier', 'n', 'beta', 'alfa', 'K1', 'K2', 'gamma1', 'gamma2', 'split'])  # Agregar columna 'split' y agregar las derivadas

# Inicializar lista para almacenar datos
data_buffer = np.array([]).reshape(0,11)

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
        initial_guesses = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        solutions = []
        for initial_guess in initial_guesses:
            solution = fsolve(equations, initial_guess, args=(beta, n, K1, K2, gamma1, gamma2, alfa))
            solutions.append(solution)

        # Calcular la distancia entre las soluciones
        distances = np.zeros((len(solutions), len(solutions)))

        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                distance = np.linalg.norm(np.array(solutions[i]) - np.array(solutions[j]))
                distances[i, j] = distance
        
        threshold = 1e-6

        # Encontrar las soluciones cercanas y calcular los puntos medios
        close_solutions = set()
        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                if distances[i, j] < threshold:
                    point1 = solutions[i]
                    point2 = solutions[j]
                    midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
                    close_solutions.add([midpoint])
                else:
                    point = solutions[i]
                    close_solutions.add([point])

        close_solutions = list(close_solutions)

        # Calcular las derivadas para todos los valores en solutions
        du_dt_values = []
        dv_dt_values = []

        for solution in close_solutions:
            du_dt, dv_dt = calculate_rates(solution, beta, n, K1, K2, gamma1, gamma2, alfa)
            du_dt_values.append(du_dt)
            dv_dt_values.append(dv_dt)

        identifier = 'i'
        if all(abs(du_dt_values) < threshold) and all(abs(dv_dt_values) < threshold):
          eigenvalues = calculate_eigenvalues(solutions, beta, n, K1, K2, gamma1, gamma2, alfa)
          if all(np.real(eigenvalues) < 0):
            identifier = 'e'

        for solution in close_solutions:
            # Agregar datos al búfer con el identificador de conjunto y el identificador de estabilidad
            data_entry = np.array(solution[0], solution[1], identifier, n, beta, K1, K2, gamma1, gamma2, alfa, 0])  # 0 para conjunto de entrenamiento
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
            data_buffer = np.array([]).reshape(0,11) # Vaciar Lista

    # Si hay datos restantes en el búfer, escribirlos en el archivo CSV
    if data_buffer.shape[0] > 0:
        if len(X_temp) >= 2:
            # Dividir datos en conjuntos de entrenamiento, validación y prueba
            X_train, X_temp, y_train, y_temp = train_test_split(data_buffer[:, :-1], data_buffer[:, :2], test_size=0.1, random_state=42)
            if len(X_temp) >= 2:
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
            else:
                print("No hay suficientes datos para la división de conjuntos.")
        else:
            print("No hay suficientes datos para la división de conjuntos.")

# Ruta de la carpeta en Google Drive donde deseas guardar el archivo CSV
google_drive_folder = '/content/drive/MyDrive/Colab Notebooks/Archivos/'

# Crear archivo CSV para almacenar datos
csv_filename = os.path.join(google_drive_folder, 'soluciones_Toggle_switch.csv')

df = pd.read_csv(csv_filename)
# filtered_df = df[(df['du_dt'] == 0) & (df['dv_dt'] == 0)]


# Contar la cantidad de datos por etiqueta
counts = df['split'].value_counts()
# counts_filter = filtered_df['split'].value_counts()
# counts_filter_n = filtered_df['n'].value_counts()

# Imprimir los resultados
print("Cantidad de datos generados para entrenamiento:", counts[0])
print("Cantidad de datos generados para validacion:", counts[1])
print("Cantidad de datos generados para prueba:", counts[2])
print("Dataframe:",df)
print("Generación y escritura de datos completadas.")

# print("Cantidad de datos verificados para entrenamiento:", counts_filter[0])
# print("Cantidad de datos verificados para validacion:", counts_filter[1])
# print("Cantidad de datos verificados para prueba:", counts_filter[2])
# print("Generación y escritura de datos completadas.")

# print("valores para n=2:",counts_filter_n[2])
# print("valores para n=3:",counts_filter_n[3])
# print("valores para n=4:",counts_filter_n[4])


# Definir una cuadrícula de puntos en el espacio fase
u = np.linspace(0, 2, 20)  # Valores de u
v = np.linspace(0, 2, 20)  # Valores de v

# Crear una cuadrícula de coordenadas para todos los puntos
U, V = np.meshgrid(u, v)

# Calcular las tasas de cambio (du_dt y dv_dt) para cada punto de la cuadrícula
du_dt, dv_dt = calculate_rates([U, V], beta, n, K1, K2, gamma1, gamma2, alfa)

# Crear el gráfico del campo vectorial
plt.figure(figsize=(8, 8))
plt.quiver(U, V, du_dt, dv_dt, scale=3.1, color='b', alpha=0.6)
plt.xlabel('u')
plt.ylabel('v')
plt.title('Campo Vectorial en el Espacio Fase')
plt.show()

# Definir una cuadrícula de puntos en el espacio fase
u = np.linspace(0, 2, 20)  # Valores de u
v = np.linspace(0, 2, 20)  # Valores de v

# Crear una cuadrícula de coordenadas para todos los puntos
U, V = np.meshgrid(u, v)

# Normalizar la punta de las flechas
du_dt_norm = du_dt/np.sqrt((du_dt**2 + dv_dt**2))
dv_dt_norm = dv_dt/np.sqrt((du_dt**2 + dv_dt**2))

# Crear el gráfico del campo vectorial
plt.figure(figsize=(8, 8))
plt.quiver(U, V, du_dt_norm, dv_dt_norm, scale=50, color='b', alpha=0.6)
plt.xlabel('u')
plt.ylabel('v')
plt.title('Campo Vectorial en el Espacio Fase')
plt.show()
