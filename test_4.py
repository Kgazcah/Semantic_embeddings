import pandas as pd

# === 1. Cargar el dataset ===
df = pd.read_csv("spam_ham_dataset.csv")

# === 2. Limpiar los textos: eliminar saltos de línea ===
df["text"] = df["text"].replace(r"\s+", " ", regex=True).str.strip()

# === 3. Separar spam y ham ===
spam = df[df["label"] == "spam"]
ham = df[df["label"] == "ham"]

# === 4. Tomar la misma cantidad de ham que de spam ===
ham_sample = ham.sample(n=len(spam), random_state=42)

# === 5. Combinar ambas partes y mezclar ===
balanced_df = pd.concat([spam, ham_sample]).sample(frac=1, random_state=42)

# === 6. Conservar sólo las columnas necesarias ===
balanced_df = balanced_df[["text", "label_num"]]

# === 7. Guardar a CSV sin saltos de línea ===
balanced_df.to_csv("spam_ham_balanced.csv", index=False, encoding="utf-8", line_terminator="\n")

print("✅ Archivo guardado como spam_ham_balanced.csv")
