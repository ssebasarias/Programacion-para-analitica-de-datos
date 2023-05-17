1. Regresión lineal simple para predecir el peso de los peces en función de su longitud vertical:

La regresión lineal simple es un modelo estadístico que se utiliza para analizar la relación entre dos variables continuas, en este caso, la longitud vertical de los peces y su peso. En esta regresión, la longitud vertical es la única variable predictora, lo que significa que estamos tratando de predecir el peso de los peces utilizando solo esa variable. Después de ajustar el modelo, podemos hacer una predicción del peso de un pez con una longitud vertical determinada.

2. Regresión lineal múltiple para predecir el peso de los peces en función de su longitud vertical y su anchura transversal:

La regresión lineal múltiple es similar a la regresión lineal simple, pero en lugar de tener una sola variable predictora, tenemos varias. En este caso, estamos utilizando la longitud vertical y la anchura transversal de los peces como variables predictoras. Esto significa que estamos tratando de predecir el peso de los peces utilizando estas dos variables. Después de ajustar el modelo, podemos hacer una predicción del peso de un pez con una longitud vertical y una anchura transversal determinadas.

3. Regresión lineal múltiple para predecir el peso de los peces en función de su longitud vertical, su anchura transversal y su especie:

Esta regresión lineal múltiple utiliza tres variables predictoras: la longitud vertical, la anchura transversal y la especie del pez. Sin embargo, la especie no es una variable numérica, sino categórica. Para incluir esta variable en el modelo, necesitamos convertirla en variables dummy utilizando un codificador one-hot. Esto significa que creamos una columna para cada categoría de la variable y ponemos un 1 en la columna correspondiente a la categoría del pez y un 0 en todas las demás columnas. Después de ajustar el modelo con las variables predictoras codificadas, podemos hacer una predicción del peso de un pez con una longitud vertical, una anchura transversal y una especie determinadas.

Cada regresión lineal tiene diferentes variables predictoras y se pueden utilizar en diferentes situaciones, dependiendo de los datos disponibles y el objetivo de la prediccion.
__________________________
df.describe(): Este método devuelve una tabla que contiene estadísticas descriptivas para todas las columnas numéricas de un DataFrame (como media, mediana, desviación estándar, valores mínimo y máximo, etc.). La salida es útil para tener una idea general de la distribución de los datos en el conjunto de datos.

df["Species"].value_counts(): Este método cuenta cuántas veces aparece cada valor único en una columna específica del DataFrame. En este caso, estamos contando cuántos ejemplos hay para cada especie de pez en la columna "Species". El resultado es útil para entender mejor cómo los diferentes tipos de peces se distribuyen a lo largo del conjunto de datos.

df.corr(): Este método calcula la correlación entre pares de variables en un DataFrame, utilizando el coeficiente de correlación de Pearson (que mide la fuerza de una relación lineal entre dos variables). La matriz resultante muestra cómo cada variable se correlaciona con todas las demás variables. Este resultado puede ayudar a identificar qué variables están más relacionadas y podría ser útil para identificar patrones interesantes dentro del conjunto de datos.

__________________

df.describe() es un método que proporciona una descripción estadística de un DataFrame. El método calcula y devuelve varias medidas estadísticas descriptivas que resumen la tendencia central, la dispersión y la forma de distribución del conjunto de datos.

Las estadísticas que se incluyen en el resultado de este método son las siguientes:

Conteo: número de observaciones no nulas en cada columna del DataFrame.
Media: promedio aritmético de los valores de cada columna.
Desviación estándar: medida de la variabilidad o dispersión alrededor de la media de cada columna. Indica cuánto se alejan los valores típicos (no extremos) de la media.
Mínimo: valor más pequeño de cada columna.
Cuartiles (25%, 50%, 75%): valores que dividen los datos ordenados en cuatro partes iguales. El primer cuartil (25%) indica el valor por debajo del cual cae el 25% de los datos; la mediana (50%) indica el valor por debajo del cual se encuentra el 50% de los datos, mientras que el tercer cuartil (75%) indica el valor por debajo del cual se encuentra el 75% de los datos.
Máximo: valor más grande de cada columna.
La salida del método "describe" puede ayudarte a entender mejor la distribución de tus datos, a identificar valores atípicos y a comparar diferentes grupos de datos. Por ejemplo, si tienes un conjunto de datos que contiene información financiera sobre diferentes empresas, puedes utilizar este método para obtener estadísticas como el promedio de ingresos, la variabilidad de los ingresos y los máximos y mínimos de cada categoría para comparar el rendimiento financiero de las diferentes empresas.
___________________

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Import the dataset
df = pd.read_csv("fish_market.csv")

# Create a scatterplot of the data
plt.scatter(df["length"], df["weight"])

# Fit a linear regression line to the data
reg = LinearRegression()
reg.fit(df[["length"]], df["weight"])

# Calculate the R-squared value
r_squared = reg.score(df[["length"]], df["weight"])

# Predict the weight of a fish given its length and price
length = 10
price = 5
weight = reg.predict([[length, price]])

print(weight)