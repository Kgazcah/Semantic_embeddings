from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
from autoencoder.nn import Autoencoder
from preprocessing.interface_builder import Builder
from preprocessing.director import Director
from preprocessing.preprocessor_builder import Preprocessing
from vocabulary.getting_vocabulary import GettingVocabulary
from encoding.lambda_grams import LambdaGrams
from encoding.lambda_grams_to_indx import LambdaGramsToIndx
from encoding.binary_embeddings import LambdaGramEmbeddings

# #split the original dataset to train same X, same Y
# def split_data (df_folder, output_folder, test_size=0.20, random_state=42, column='embedding', semantic=False):
#     df = pd.read_csv(df_folder, dtype={column: str})
#     if semantic:

#     X_train, X_test, y_train, y_test = train_test_split(
#     df, df, test_size=test_size, random_state=random_state)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.20, random_state=42)

#     #as 'y' data must be the same as 'X' data we do not save it in all cases
#     X_train = pd.DataFrame(X_train)
#     X_train.to_csv(f'{output_folder}/X_train.csv', index=False)

#     X_test = pd.DataFrame(X_test)
#     X_test.to_csv(f'{output_folder}/X_test.csv', index=False)

#     X_val = pd.DataFrame(X_val)
#     X_val.to_csv(f'{output_folder}/X_val.csv', index=False)

def split_data(df, output_folder, binary_embeddings_file,
               test_size=0.20, random_state=42, n_gram=1, semantic=False):

    emb_df = pd.read_csv(binary_embeddings_file, dtype={'embedding': str})
    emb_df['lambda_gram'] = emb_df['lambda_gram'].astype(str).str.strip()

    if not semantic:
        X_train, X_test, y_train, y_test = train_test_split(
            df, df, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=random_state)

        X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
        X_val.to_csv(f'{output_folder}/X_val.csv', index=False)
        X_test.to_csv(f'{output_folder}/X_test.csv', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        ngram_to_emb = dict(zip(emb_df['lambda_gram'], emb_df['embedding']))
        pairs = []
        skipped = 0

        for _, row in df.iterrows():
            text = str(row['text'])
            tokens = text.split()
            n = n_gram

            if len(tokens) <= n:
                continue

            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

            for i in range(len(ngrams) - 1):
                x_gram = ngrams[i]
                y_gram = ngrams[i+1]

                if x_gram in ngram_to_emb and y_gram in ngram_to_emb:
                    pairs.append((x_gram, ngram_to_emb[x_gram],
                                  y_gram, ngram_to_emb[y_gram]))
                else:
                    skipped += 1

        df_pairs = pd.DataFrame(pairs, columns=['X_text', 'X_emb', 'y_text', 'y_emb'])

        df_pairs.to_csv(f'{output_folder}/all_pairs.csv', index=False)

        print(f"Total pairs: {len(df_pairs)}")
        print(f"Missing pairs: {skipped}")

        X_train, X_test, y_train, y_test = train_test_split(
            df_pairs[['X_text', 'X_emb']], df_pairs[['y_text', 'y_emb']],
            test_size=test_size, random_state=random_state)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=random_state)

        X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
        X_val.to_csv(f'{output_folder}/X_val.csv', index=False)
        X_test.to_csv(f'{output_folder}/X_test.csv', index=False)
        y_train.to_csv(f'{output_folder}/y_train.csv', index=False)
        y_val.to_csv(f'{output_folder}/y_val.csv', index=False)
        y_test.to_csv(f'{output_folder}/y_test.csv', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test


#preprocessing the corpus (Builder Pattern)
def preprocessing(df, type, language='en'):
    director = Director()
    builder = Preprocessing(df)
    if type == 'basic':
        #basic builder includes stopwords
        preprocessed_df = director.makeBasicPreprocessing(builder, language)
    elif type == 'plus':
        #plus builder does not include stopwords
        preprocessed_df = director.makePlusPreprocessing(builder, language)
    return preprocessed_df



#Getting the vocabulary and their indexes
def get_vocab_ind_bin(preprocessed_df, output_file='assets/method', code='bcd'):
    vocab_obj = GettingVocabulary(preprocessed_df)
    vocabulary = vocab_obj.get_vocab()
    vocab = pd.DataFrame(vocabulary)
    vocab.to_csv(f'{output_file}/vocabulary.csv', index=False)

    #assign a decimal index to each word from the vocabulary
    #vocabulary to index document will have something as follows:
    #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
    vocab_to_index = vocab_obj.get_vocab_to_indx()
    vocab_to_index_df = pd.DataFrame(list(vocab_to_index.items()), 
                                    columns=['word', 'index'])
    vocab_to_index_df.to_csv(f'{output_file}/vocab_to_index.csv', index=False)

    # If you want to upload your own vocabulary to index, uncomment the next 3 lines.
    # The dictionary must be in the form: #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
    # vocab_to_index_df = pd.read_csv('assets/vocab_to_index.csv')
    # columns = ['word', 'index']
    # vocab_to_index = vocab_to_index_df.set_index(columns[0])[columns[1]].to_dict()

    #get the binary embedding from the decimal indexes
    vocab_to_binary, bits = vocab_obj.get_binary_rep(vocab_to_index, code=code)

    #the embedding binary dictionary has the following example form:
    # {'yet': '110000',..., 'zero': '100001'}
    binary_dic = pd.DataFrame(list(vocab_to_binary.items()), columns=['word', 'binary'])
    binary_dic.to_csv(f'{output_file}/vocab_to_binary.csv', index=False)
    return vocab_to_binary, vocab_to_index, bits

#Getting lambda grams
def get_lambda_grams(preprocessed_df, n_gram, classify=False):
    n_grams = LambdaGrams(preprocessed_df)
    if classify:
        lambda_grams = n_grams.get_lambda_grams_for_classify(n_gram)
    else:
        #for 3 gram example we will have something as follows: 
        #  ['carried distributed manner']
        lambda_grams = n_grams.get_lambda_grams(n_gram)
    return lambda_grams

# Lambda grams to indexes
def lambda_grams_to_indexes(lambda_grams, vocab_to_index):
    #convert the lambda grams to indexes (e.g., [259, 533, 1060] )
    #for this we need the vocabulary to index document of the form:
    #{'x': 1, ..., 'yet': 1984, 'zero': 1985}

    lti = LambdaGramsToIndx(lambda_grams, vocab_to_index)
    lambda_gram_to_index = lti.get_lambda_grams_indx()


#Encoding the lambda grams
def lambda_grams_to_binary(vocab_to_binary, lambda_grams, output_file_name, fun=0):
    binary_encode_decode = LambdaGramEmbeddings(vocab_to_binary)
    dictionary = vocab_to_binary
    #saving the binary embeddings
    binary_lambda_grams = binary_encode_decode.get_embeddings_df(lambda_grams, fun=fun)
    binary_lambda_grams.to_csv(output_file_name, index=False)
    return dictionary

def lambda_grams_to_binary_for_classify(vocab_to_binary, lambda_grams):
    binary_encode_decode = LambdaGramEmbeddings(vocab_to_binary)
    binary_lambda_grams = binary_encode_decode.get_embeddings_df_to_classify(lambda_grams)
    return binary_lambda_grams
    

def binary_to_ngrams(binary_embedding, ind, n_gram, vocab_to_binary, bits=None, code='bcd'):
    binary_encode_decode = LambdaGramEmbeddings(vocab_to_binary)
    return binary_encode_decode.binary_to_ngrams(binary_embedding[ind], n_gram, vocab_to_binary, bits=bits,code=code)


#Uploading the data
def upload_data_to_train(file_name, column):
    df = pd.read_csv(file_name, dtype={"embedding": str})
    return np.vstack(df[column].apply(lambda x: np.array(list(x), dtype=float)).values)

def upload_vocab_to_binary_dictionary(file='binary_dict_karina.csv', columns=['word', 'binary']):
    # The dictionary must be in the form: {'yet': '110000', 'zero': '100001'}
    df = pd.read_csv(file, dtype={'binary': str})
    dictionary = df.set_index(columns[0])[columns[1]].to_dict()
    return dictionary

def generate_document_embeddings(n_grams, embeddings, sentences):
    pass




# to do: adding the following as a function

# import numpy as np
# from autoencoder.nn import Autoencoder

# yuri_embeddings = np.loadtxt('assets/n_gram_embeddings/yuri_n_gram_embedding.tsv', delimiter='\t')
# kari_embeddings = np.loadtxt('assets/n_gram_embeddings/n_gram_embedding.tsv', delimiter='\t')

# autoencoder = Autoencoder()
# model= autoencoder.load_model()
# decode = autoencoder.decode()

# yuri_binary = np.round(decode.predict(yuri_embeddings)).astype(int)
# kari_binary = np.round(decode.predict(kari_embeddings)).astype(int)

# diffs = kari_binary != yuri_binary 
# false_counts = np.sum(diffs, axis=1)

# equal_indices = np.where(false_counts == 0)[0] 
# equal_count = len(equal_indices)
# different_sum = np.sum(false_counts)

# print(f"Equal: {equal_count}")
# print(f"Average differences: {different_sum / len(kari_binary):.2f}")
# print(f"Equal indexes: {equal_indices.tolist()}")
 




#to do adding this in a function to eliminate the rows with less than 3 n grams
# df = pd.read_csv('data/sotware_requirements/dataset.csv')
# preprocessed_df = utils.preprocessing(df, 'plus')
# df['preprocessed'] = preprocessed_df
# df_filtrado = df[df['preprocessed'].str.split().str.len() >= 3].reset_index(drop=True)
# print(df_filtrado)
# df_filtrado = df_filtrado.drop(columns=['preprocessed'])
# df_filtrado.to_csv('assets/dataset.csv', index=False)

#to do: integrate embedding function in utils