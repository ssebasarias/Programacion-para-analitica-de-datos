# Importamos las librerías necesarias
import pandas as pd
import numpy as np

# Leemos el archivo csv desde una URL
url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
df = pd.read_csv(url)

# Reemplazamos los valores nulos por la media de la columna correspondiente
df.fillna(df.mean(), inplace=True)

# Normalizamos los nombres de las columnas y los valores de las variables categóricas
df.columns = df.columns.str.lower().str.replace(" ", "_")
df["name"] = df["name"].str.lower()
df["type_1"] = df["type_1"].str.lower()
df["type_2"] = df["type_2"].str.lower()

# Creamos una nueva columna con el total de estadísticas
df["total_stats"] = df["hp"] + df["attack"] + df["defense"] + df["sp._atk"] + df["sp._def"] + df["speed"]

# Eliminamos las columnas que no aportan información relevante o que son redundantes
df.drop(columns=["#", "generation", "legendary"], inplace=True)

# Agrupamos el dataframe por tipo de pokemon y calculamos la media de cada grupo
df_grouped = df.groupby("type_1").mean()

# Codificamos las variables categóricas con números
type_dict = {t: i for i, t in enumerate(df["type_1"].unique())}
df["type_1_code"] = df["type_1"].map(type_dict)