import pandas as pd

# Cargar el archivo CSV que contiene tus datos
csv_filename = 'datos_puntos_estacionarios.csv'
df = pd.read_csv(csv_filename)

# Contar la cantidad de datos por etiqueta
counts = df['split'].value_counts()

# Imprimir los resultados
print("Cantidad de datos con etiqueta 0:", counts[0])
print("Cantidad de datos con etiqueta 1:", counts[1])
print("Cantidad de datos con etiqueta 2:", counts[2])
