import pandas as pd
import numpy as np

# Cargar el DataFrame
df = pd.read_csv('data/software_requirements/no_stopwords/dataset_ngrams.csv')

# Convertir los embeddings de string a listas numéricas
def parse_embedding(emb_str):
    # Quitar corchetes y dividir por espacio
    emb_str = emb_str.replace('[', '').replace(']', '').strip()
    # Convertir a lista de floats
    return [float(x) for x in emb_str.split() if x.strip() != '']

# Aplicar la función a cada fila
embeddings = df['4_gray_gram_embeddings'].apply(parse_embedding)

# Guardar embeddings en formato .tsv (valores separados por tabulador, sin comillas ni corchetes)
with open('sentences_embeddings.tsv', 'w', encoding='utf-8') as f:
    for emb in embeddings:
        f.write('\t'.join(map(str, emb)) + '\n')

# Guardar textos en otro archivo .tsv
df['text'].to_csv('sentences.tsv', sep='\t', index=False, header=False)
