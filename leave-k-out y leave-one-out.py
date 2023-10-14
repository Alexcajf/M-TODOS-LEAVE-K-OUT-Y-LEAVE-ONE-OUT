import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
data = np.genfromtxt("irisbin.csv", delimiter=',')
X = data[:, :4]  # Características (dimensiones de pétalos y sépalos)
y = data[:, 4:]  # Etiquetas (código binario de la especie)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear un perceptrón multicapa
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Entrenar la red neuronal
mlp.fit(X_train, y_train)

# Evaluar el rendimiento en el conjunto de prueba
accuracy = mlp.score(X_test, y_test)
print("Exactitud en el conjunto de prueba:", accuracy)

# Validación cruzada Leave-One-Out
loo = cross_val_score(mlp, X, y, cv=len(X))

# Calcular el error esperado de clasificación
error_loo = 1 - np.mean(loo)
std_dev_loo = np.std(loo)

print("Error esperado de clasificación (Leave-One-Out):", error_loo)
print("Desviación estándar (Leave-One-Out):", std_dev_loo)

# Gráfico de barras para visualizar el rendimiento de validación cruzada Leave-One-Out
fig, ax = plt.subplots()
ax.bar(['Error Esperado', 'Desviación Estándar'], [error_loo, std_dev_loo])
ax.set_ylabel('Valor')
ax.set_title('Rendimiento de Validación Cruzada Leave-One-Out')
plt.show()
