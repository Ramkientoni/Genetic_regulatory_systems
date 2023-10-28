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
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Función para resolver el sistema de ecuaciones
def equations(x, beta, n, K1, K2, gamma1, gamma2, alfa):
    u, v = x
    du_dt = (beta / (1 + (v/K1) ** n)) - gamma1*u
    dv_dt = (alfa / (1 + (u/K2) ** n)) - gamma2*v
    return (du_dt, dv_dt)

# Calcular distancias
def distancia_entre_puntos(punto1, punto2):
    return np.linalg.norm(np.array(punto1) - np.array(punto2))

# Agrupar valores cercanos
def agrupar_por_umbral(solutions, umbral):
    grupos = []
    for p1 in solutions:
        for p2 in solutions:
            if (p1 == p2):
                continue
            dist = distancia_entre_puntos(p1, p2)
            if dist < umbral:
                needToAdd = True
                for g in grupos:
                    if p1 in g and p2 not in g:
                        g.append(p2)
                        needToAdd = False
                    elif p2 in g and p1 not in g:
                        g.append(p1)
                        needToAdd = False
                    elif p1 in g and p2 in g:
                        needToAdd = False
                    elif p1 not in g and p2 not in g:
                        needToAdd = True

                if needToAdd:
                    grupos.append([p1, p2])

    for p1 in solutions:
        esta = False
        for g in grupos:
            if p1 in g:
                esta = True
        if not esta:
            grupos.append([p1])

    return grupos

# Obtener puntos medios
def promediar_grupos(grupos):
    grupos_medios = []
    for g in grupos:
        s = [0, 0]
        for p in g:
            s[0] += p[0]
            s[1] += p[1]
        s[0] /= len(g)
        s[1] /= len(g)
        grupos_medios.append(s)
    return grupos_medios

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

num_iterations = 100000  # Número de iteraciones para generar datos
chunk_size = 1000  # Tamaño del chunk

K1_values = np.random.uniform(0, 1, num_iterations)
K2_values = np.random.uniform(0, 1, num_iterations)
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
csv_filename = os.path.join(google_drive_folder, 'puntos_estacionarios_Toggle_switch.csv')

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
    initial_guesses = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    solutions = []
    for initial_guess in initial_guesses:
      solution = tuple(fsolve(equations, initial_guess, args=(beta, n, K1, K2, gamma1, gamma2, alfa)))
      solutions.append(solution)

    # Agrupar los puntos cercanos entre si
    grupos = agrupar_por_umbral(solutions, 1e-3)
    puntos_estacionarios = promediar_grupos(grupos)

    # Calcular las derivadas para todos los valores en solutions
    for solution in puntos_estacionarios:

      identifier = 'i'

      du_dt, dv_dt = calculate_rates(solution, beta, n, K1, K2, gamma1, gamma2, alfa)

      if du_dt < 1e-3 and dv_dt < 1e-3:
        eigenvalues = calculate_eigenvalues(solution, beta, n, K1, K2, gamma1, gamma2, alfa)
        if all(eigenvalues < 0): #agregar identificador para puntos estables: e si no se cumple la condicion: i
          identifier = 'e'

      # Agregar datos al búfer con el identificador de conjunto
      data_entry = np.array([solution[0], solution[1], identifier, n, beta, K1, K2, gamma1, gamma2, alfa, 0])  # 0 para conjunto de entrenamiento
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
csv_filename = os.path.join(google_drive_folder, 'puntos_estacionarios_Toggle_switch.csv')

df = pd.read_csv(csv_filename)

# Contar la cantidad de datos por etiqueta
counts = df['split'].value_counts()
estabilidad = df['stability identifier'].value_counts()

# Imprimir los resultados
print("Cantidad de datos generados para entrenamiento:", counts[0])
print("Cantidad de datos generados para validacion:", counts[1])
print("Cantidad de datos generados para prueba:", counts[2])
print("Cantidad de puntos estables:", estabilidad["e"])
print("Cantidad de puntos inestables:", estabilidad["i"])
print("Dataframe:",df)





