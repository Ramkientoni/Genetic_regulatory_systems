import numpy as np
from scipy.optimize import fsolve
import csv

# Definición de las ecuaciones del sistema


def equations(x, beta, n, K, gamma, alfa):
    u, v = x
    du_dt = (beta / (1 + (v/K) ** n)) - gamma*u
    dv_dt = (alfa / (1 + (u/K) ** n)) - gamma*v
    return [du_dt, dv_dt]


# Número de iteraciones
num_iterations = 1000000

# Nombre del archivo CSV para guardar los datos
csv_filename = 'datos_puntos_estacionarios.csv'

# Guardar la información en el archivo CSV
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Escribir la cabecera del archivo CSV
    writer.writerow(['n', 'beta', 'K', 'gamma', 'alfa', 'u', 'v'])

    for _ in range(num_iterations):

        # Generar parámetros aleatorios en el rango [0, 1]
        K = np.random.uniform(0, 1)
        beta = np.random.uniform(0, 1)
        gamma = np.random.uniform(0, 1)
        alfa = np.random.uniform(0, 1)
        n = 2

        # Encontrar los puntos estacionarios para los parámetros aleatorios
        initial_guess = [0.0, 0.0]
        u, v = fsolve(
            equations, initial_guess, args=(beta, n, K, gamma, alfa))

        writer.writerow([n, beta, K, gamma, alfa, u, v])

print(f"Datos guardados en '{csv_filename}'")
