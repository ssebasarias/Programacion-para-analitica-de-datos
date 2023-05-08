# Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# Leer el data set
df = pd.read_csv("Fish.csv")

# Verificar el tipo de datos, el número de filas y columnas, y si hay valores nulos o duplicados
df.info()
df.shape
df.isnull().sum()
df.duplicated().sum()

# Limpiar y transformar los datos (en este caso no hay mucho que hacer, solo cambiar el nombre de una columna)
df = df.rename(columns={"Length1":"Length_Vertical"})

# Explorar los datos con estadísticas descriptivas y gráficos
df.describe()
df["Species"].value_counts()
df.corr()

# Hacer un histograma del peso de los peces
# muestra la distribución del peso de los peces
plt.hist(df["Weight"], bins=20) 
plt.xlabel("Weight (g)")
plt.ylabel("Frequency")
plt.title("Histogram of Fish Weight")
plt.show()

# Hacer un boxplot del peso de los peces según la especie
# muestra cómo varía el peso con cada especie de pez
sns.boxplot(x="Species", y="Weight", data=df) 
plt.xlabel("Species")
plt.ylabel("Weight (g)")
plt.title("Boxplot of Fish Weight by Species")
plt.show()

# Hacer un gráfico de dispersión del peso y la longitud vertical de los peces
# muestra si hay alguna relación entre el peso y la longitud vertical de los peces
plt.scatter(df["Weight"], df["Length_Vertical"])
plt.xlabel("Weight (g)")
plt.ylabel("Length Vertical (cm)")
plt.title("Scatterplot of Fish Weight and Length Vertical")
plt.show()

# Hacer un mapa de calor de la matriz de correlación entre las variables numéricas
# muestra las correlaciones entre todas las variables numéricas
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Heatmap of Correlation Matrix")
plt.show()

# Regresion lineal simple: Predecir el peso de los peces en funcion a su longitud:
from sklearn.linear_model import LinearRegression

# Seleccionar las variables predictoras y la variable objetivo
X = df[["Length_Vertical"]]
y = df["Weight"]

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(X, y)

# Hacer una predicción con una longitud vertical de 25 cm
prediction = model.predict([[25]])

# Imprimir la predicción
print(f"La predicción del modelo para un pez con una longitud vertical de 25 cm es {prediction[0]:.2f} gramos.")


# Regresion lineal multiple: Predecir el peso de los peces en funcion a su longitud vertical y anchuratransversal:
from sklearn.linear_model import LinearRegression

# Seleccionar las variables predictoras y la variable objetivo
X = df[["Length_Vertical", "Width"]]
y = df["Weight"]

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(X, y)

# Hacer una predicción con una longitud vertical de 25 cm y una anchura transversal de 10 cm
prediction = model.predict([[25, 10]])

# Imprimir la predicción
print(f"La predicción del modelo para un pez con una longitud vertical de 25 cm y una anchura transversal de 10 cm es {prediction[0]:.2f} gramos.")


# Regresion lineal multiple: Predecir el peso de los peces en función de su longitud vertical, anchura transversal y su especie:
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Seleccionar las variables predictoras y la variable objetivo
X = df[["Length_Vertical", "Width", "Species"]]
y = df["Weight"]

# Crear un codificador one-hot para la variable categórica 'Species'
encoder = OneHotEncoder()
ct = ColumnTransformer([("encoder", encoder, [2])], remainder="passthrough")
X_encoded = ct.fit_transform(X)

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(X_encoded, y)

# Hacer una predicción para un Bass con una longitud vertical de 25 cm y una anchura transversal de 10 cm
prediction = model.predict([[1, 0, 0, 25, 10]])

# Imprimir la predicción
print(f"La predicción del modelo para un Bass con una longitud vertical de 25 cm y una anchura transversal de 10 cm es {prediction[0]:.2f} gramos.")
