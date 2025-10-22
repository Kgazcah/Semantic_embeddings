import pandas as pd
import numpy as np
from autoencoder.nn import Autoencoder
import utils
import ast


n_gram = '4_gray'
bin_word_size = 11
n_gram_n = 4
vocab_size = 1979
binary_embedding_size = bin_word_size * int(n_gram_n)
problem = 'software_requirements/no_stopwords'

df = pd.read_csv(f'data/{problem}/dataset_ngrams.csv')

######################## Adding the lambda grams
# preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic')
else:
    preprocessed_df = utils.preprocessing(df, 'plus')

# construct the lambda grams for each sentence
l_grams = utils.get_lambda_grams(preprocessed_df, n_gram_n, classify=True)

l_grams_df = pd.DataFrame({'n_grams': l_grams})
df[f'{n_gram}_grams'] = l_grams_df
print(df.head())
df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)

######################## Adding lambda grams to binary representations 

l_grams = df[f'{n_gram}_grams'].to_list()
dictionary = utils.upload_vocab_to_binary_dictionary(file=f'assets/method/{problem}/vocab_to_binary.csv')

sentence_binary_embeddings = []
for lg in l_grams:
    if isinstance(lg, str):
        lg = ast.literal_eval(lg)
    binary_embeddings = utils.lambda_grams_to_binary_for_classify(dictionary, lg)
    sentence_binary_embeddings.append(binary_embeddings)

binary_df = pd.DataFrame({f'{n_gram}_gram_binary': sentence_binary_embeddings})
df[f'{n_gram}_gram_binary'] = binary_df
df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)

######################### Encoding the binary embeddings

def prepare_data(binary_embedding):
    return [float(d) for d in binary_embedding]

def fix_vector_length(vec, target_len):
    """Recorta o rellena con ceros para que todos los vectores tengan la misma longitud."""
    if len(vec) > target_len:
        return vec[:target_len]
    elif len(vec) < target_len:
        return vec + [0.0] * (target_len - len(vec))
    return vec


autoencoder = Autoencoder(input_size=binary_embedding_size, input_neurons=binary_embedding_size, vocab_size=vocab_size, n_gram=n_gram_n, bits_per_token=bin_word_size)
model = autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5', vocab_size=vocab_size, bits_per_token=bin_word_size)
encode = autoencoder.encode()

bin_grams = df[f'{n_gram}_gram_binary'].to_list()
embedding_dim = 200  

sentences_embeddings = []

for bg in bin_grams:
    if isinstance(bg, str):
        bg_list = ast.literal_eval(bg)
    else:
        bg_list = bg

    if len(bg_list) == 0:
        # if empty generate an embedding (200dim)
        n_grams_embeddings = np.zeros((1, embedding_dim))
    else:
        bg_prepared = [prepare_data(x) for x in bg_list]

        # asegurar longitud fija en todos los vectores
        bg_prepared = [fix_vector_length(vec, binary_embedding_size) for vec in bg_prepared]

        # ahora sí se pueden apilar
        bg_prepared = np.vstack(bg_prepared).astype(float)

        if len(bg_prepared.shape) == 1:
            bg_prepared = np.expand_dims(bg_prepared, axis=0)

        n_grams_embeddings = encode.predict(bg_prepared)

        if len(n_grams_embeddings) > 0:
            n_grams_embeddings = np.sum(n_grams_embeddings, axis=0)
        else:
            n_grams_embeddings = np.zeros(embedding_dim)

    sentences_embeddings.append(n_grams_embeddings)

embeddings_df = pd.DataFrame({f'{n_gram}_gram_embeddings': sentences_embeddings})
print(embeddings_df)
df[f'{n_gram}_gram_embeddings'] = embeddings_df
df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)

############################ Adding my embeddings to yuri's embeddings

df_train = pd.read_csv(f'data/{problem}/train_df.csv')
df_test = pd.read_csv(f'data/{problem}/test_df.csv')
df_val = pd.read_csv(f'data/{problem}/val_df.csv')

cols_to_add = ['text', '4_gray_gram_embeddings']
df_subset = df[cols_to_add]

df_train = df_train.merge(df_subset, on='text', how='left')
df_test  = df_test.merge(df_subset, on='text', how='left')
df_val   = df_val.merge(df_subset, on='text', how='left')

df_train.to_csv(f'data/{problem}/train_df.csv', index=False)
df_test.to_csv(f'data/{problem}/test_df.csv', index=False)
df_val.to_csv(f'data/{problem}/val_df.csv', index=False)

exit()
# import pandas as pd
# import numpy as np
# from autoencoder.nn import Autoencoder
# import utils
# import ast


# n_gram = '1_gray'
# # bcd=True
# bin_word_size = 12
# n_gram_n = 1

# binary_embedding_size = bin_word_size*int(n_gram_n)
# problem = 'software_requirements/no_stopwords'

# df = pd.read_csv(f'data/{problem}/dataset_ngrams.csv')

# ######################## Adding the lambda grams
# # preprocessing the dataset
# if problem.split('/')[1] == 'stopwords':
#     preprocessed_df = utils.preprocessing(df, 'basic')
# else:
#     preprocessed_df = utils.preprocessing(df, 'plus')
# #construct the lambda grams for each sentence
# l_grams = utils.get_lambda_grams(preprocessed_df, 4, classify=True)

# l_grams_df = pd.DataFrame({'n_grams': l_grams})
# # df['class'] = df['class'].map({'nfr': 1, 'fr': 0})
# df[f'{n_gram}_grams'] = l_grams_df
# print(df.head())
# df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)
# ######################## Adding lambda grams to binary representations 

# l_grams = df[f'{n_gram}_grams'].to_list()
# dictionary = utils.upload_vocab_to_binary_dictionary(file=f'assets/method/{problem}/vocab_to_binary.csv')

# sentence_binary_embeddings = []
# for lg in l_grams:
#     if isinstance(lg, str):
#         lg = ast.literal_eval(lg)
#     binary_embeddings = utils.lambda_grams_to_binary_for_classify(dictionary, lg)
#     sentence_binary_embeddings.append(binary_embeddings)

# binary_df = pd.DataFrame({f'{n_gram}_gram_binary':sentence_binary_embeddings})
# df[f'{n_gram}_gram_binary'] = binary_df
# df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)

# ######################### Encoding the binary embeddings

# def prepare_data(binary_embedding):
#     return [float(d) for d in binary_embedding]

# autoencoder = Autoencoder(binary_embedding_size, binary_embedding_size)
# model = autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')
# encode = autoencoder.encode()

# bin_grams = df[f'{n_gram}_gram_binary'].to_list()

# embedding_dim = 200  

# sentences_embeddings = []

# for bg in bin_grams:
#     if isinstance(lg, str):
#         bg_list = ast.literal_eval(bg)
#     else:
#         bg_list = bg

#     if len(bg_list) == 0:
#         # if empty generate an embedding (200dim)
#         n_grams_embeddings = np.zeros((1, embedding_dim))
#     else:
#         bg_prepared = [prepare_data(x) for x in bg_list]

#         max_len = binary_embedding_size  
#         bg_prepared = [vec + [0.0] * (max_len - len(vec)) for vec in bg_prepared]

#         bg_prepared = np.vstack(bg_prepared).astype(float)



#         if len(bg_prepared.shape) == 1:
#             bg_prepared = np.expand_dims(bg_prepared, axis=0)

#         n_grams_embeddings = encode.predict(bg_prepared)

#         if len(n_grams_embeddings) > 0:
#             n_grams_embeddings = np.sum(n_grams_embeddings, axis=0)
#         else:
#             n_grams_embeddings = np.zeros(embedding_dim)

#     sentences_embeddings.append(n_grams_embeddings)

# embeddings_df = pd.DataFrame({f'{n_gram}_gram_embeddings': sentences_embeddings})
# print(embeddings_df)
# df[f'{n_gram}_gram_embeddings'] = embeddings_df
# df.to_csv(f'data/{problem}/dataset_ngrams.csv', index=False)


# ############################ Adding my embeddings to yuri's embeddings

# df_train = pd.read_csv(f'data/{problem}/train_df.csv')
# df_test = pd.read_csv(f'data/{problem}/test_df.csv')
# df_val = pd.read_csv(f'data/{problem}/val_df.csv')

# cols_to_add = ['text', '1_gray_gram_embeddings']

# df_subset = df[cols_to_add]

# df_train = df_train.merge(df_subset, on='text', how='left')
# df_test  = df_test.merge(df_subset, on='text', how='left')
# df_val   = df_val.merge(df_subset, on='text', how='left')

# df_train.to_csv(f'data/{problem}/train_df.csv', index=False)
# df_test.to_csv(f'data/{problem}/test_df.csv', index=False)
# df_val.to_csv(f'data/{problem}/val_df.csv', index=False)