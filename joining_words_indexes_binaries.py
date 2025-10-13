import pandas as pd

words = pd.read_csv('assets/method/software_requirements/no_stopwords/vocab_to_index.csv')
print(words)

binarios = pd.read_csv('assets/method/software_requirements/no_stopwords/vocab_to_binary.csv', dtype=str)
print(binarios)

cols_to_add = ['word', 'binary']
df_subset = binarios[cols_to_add]

df_train = words.merge(df_subset, on='word', how='left')
df_train.to_csv('1_grams.csv', index=False)