import pandas as pd

# Supongamos que tu archivo es un CSV
df = pd.read_csv("assets/method/spam/stopwords/df_preprocessed.csv")

# Nombre de la columna de texto
columna_texto = "text"

# Filtrar las filas que tengan al menos 2 palabras
df = df[df[columna_texto].str.split().str.len() >= 2]

# Verificar el resultado
print(df.head())

# (Opcional) Guardar el resultado
df.to_csv("dataset_spam.csv", index=False)
