import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, output_folder, binary_embeddings_file,
               test_size=0.20, random_state=42, n_gram=1, semantic=False):
    """
    Si semantic=False → split simple.
    Si semantic=True  → genera pares (X, y) con embeddings y guarda
                        también el archivo completo 'all_pairs.csv'.

    IMPORTANTE: El texto en df['text'] debe venir ya preprocesado.
    """

    emb_df = pd.read_csv(binary_embeddings_file, dtype={'embedding': str})
    emb_df['lambda_gram'] = emb_df['lambda_gram'].astype(str).str.strip()

    if not semantic:
        # === Split estándar (no semántico) ===
        X_train, X_test, y_train, y_test = train_test_split(
            df, df, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=random_state)

        X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
        X_val.to_csv(f'{output_folder}/X_val.csv', index=False)
        X_test.to_csv(f'{output_folder}/X_test.csv', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        # === Generación semántica con embeddings ===
        ngram_to_emb = dict(zip(emb_df['lambda_gram'], emb_df['embedding']))
        pairs = []
        skipped = 0

        for _, row in df.iterrows():
            text = str(row['text'])
            tokens = text.split()  # ya está preprocesado
            n = n_gram

            if len(tokens) <= n:
                continue

            # Crear n-gramas consecutivos
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

            # Generar pares (X, y)
            for i in range(len(ngrams) - 1):
                x_gram = ngrams[i]
                y_gram = ngrams[i+1]

                if x_gram in ngram_to_emb and y_gram in ngram_to_emb:
                    pairs.append((x_gram, ngram_to_emb[x_gram],
                                  y_gram, ngram_to_emb[y_gram]))
                else:
                    skipped += 1

        # Crear DataFrame con todos los pares
        df_pairs = pd.DataFrame(pairs, columns=['X_text', 'X_emb', 'y_text', 'y_emb'])

        # Guardar el documento completo ANTES de dividir
        df_pairs.to_csv(f'{output_folder}/all_pairs.csv', index=False)

        print(f"✅ Total de pares generados: {len(df_pairs)}")
        print(f"⚠️ Pares saltados (sin embedding): {skipped}")

        # === División en train/val/test ===
        X_train, X_test, y_train, y_test = train_test_split(
            df_pairs[['X_text', 'X_emb']], df_pairs[['y_text', 'y_emb']],
            test_size=test_size, random_state=random_state)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=random_state)

        # Guardar splits
        X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
        X_val.to_csv(f'{output_folder}/X_val.csv', index=False)
        X_test.to_csv(f'{output_folder}/X_test.csv', index=False)
        y_train.to_csv(f'{output_folder}/y_train.csv', index=False)
        y_val.to_csv(f'{output_folder}/y_val.csv', index=False)
        y_test.to_csv(f'{output_folder}/y_test.csv', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test


import utils

problem = 'software_requirements/no_stopwords'
n_gram = '1_gray'
df = pd.read_csv(f'data/{problem}/dataset.csv')
code = 'gray'
bits = 12
n_gram_n = int(n_gram[0])

# preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic', 'en')
else:
    preprocessed_df = utils.preprocessing(df, 'plus', 'en')

df_preprocessed = pd.DataFrame({'text': preprocessed_df})
df_preprocessed.to_csv('df_pre.csv', index=False)

split_data(
    df= df_preprocessed,
    output_folder='splits',
    binary_embeddings_file='assets/bin_embeddings/software_requirements/no_stopwords/1_gray_grams/binary_embeddings.csv',
    n_gram=1,
    semantic=True
)
