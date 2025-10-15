import pandas as pd
import numpy as np
from autoencoder.nn import Autoencoder
from visualization.plotting import Visualization 
import utils

problem = 'software_requirements/no_stopwords'
n_gram = '1_gray'
df = pd.read_csv(f'data/{problem}/dataset.csv')
code = 'gray'
bits = 11
n_gram_n = int(n_gram[0])
loss = 'fla'
epochs = 300

# preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic', 'en')
else:
    preprocessed_df = utils.preprocessing(df, 'plus', 'en')

df_preprocessed = pd.DataFrame({'text': preprocessed_df})
df_preprocessed.to_csv(f'assets/method/{problem}/df_tokenized.csv', index=False)    

#getting the vocabulary, vocab_to_index and vocab_to_binary
word_to_bin, vocab_to_index, bits = utils.get_vocab_ind_bin(preprocessed_df, output_file=f'assets/method/{problem}', code=code)
l_grams = utils.get_lambda_grams(preprocessed_df, n_gram_n)
vocab_size = len(word_to_bin)


#getting the lambda grams to binary embeddings
utils.lambda_grams_to_binary(word_to_bin, l_grams, f'assets/bin_embeddings/{problem}/{n_gram}_grams/binary_embeddings.csv')

################## Splitting into training and testing data 

utils.split_data(
    df = df_preprocessed,
    output_folder = f'data/{problem}/{n_gram}_grams',
    binary_embeddings_file = f'assets/bin_embeddings/software_requirements/no_stopwords/{n_gram_n}_gray_grams/binary_embeddings.csv',
    n_gram = n_gram_n,
    semantic = True
)


# utils.split_data(f'assets/bin_embeddings/{problem}/{n_gram}_grams/binary_embeddings.csv', f'data/{problem}/{n_gram}_grams')

# #################### Loading data to train autoencoder
#loading training and testing data for 4_grams
X_train = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_train.csv', 'X_emb')
X_test = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_test.csv', 'X_emb')
X_val = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/X_val.csv', 'X_emb')
# y_train = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/y_train.csv', 'y_emb')
# y_test = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/y_test.csv', 'y_emb')
# y_val = utils.upload_data_to_train(f'data/{problem}/{n_gram}_grams/y_val.csv', 'y_emb')
y_train, y_test, y_val = X_train, X_test, X_val

dictionary = utils.upload_vocab_to_binary_dictionary(file=f'assets/method/{problem}/vocab_to_binary.csv')
# # # ##################### Step 8: Creating and training the Neural Network (Autoencoder)

initialize_weights_file = f'assets/weights/{problem}/{n_gram}_grams/initial_weights.pkl'
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1], embedding_size=200, vocab_size=vocab_size, n_gram=n_gram_n, bits_per_token=bits, loss=loss)
autoencoder.save_initialize_weights(initialize_weights_file=initialize_weights_file)
history = autoencoder.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
autoencoder.save(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')

##################### Step 9: Visualizing the training plots
plot = Visualization()
plot.plotting_metric(history.history, 'cosine_similarity', 'val_cosine_similarity', path=f'assets/learning_graphs/{problem}/{n_gram}_grams', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path=f'assets/learning_graphs/{problem}/{n_gram}_grams', fig_name='Loss training')

# ################### Step 10: Predicting

# comment following line if you do not want to predict
autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1], vocab_size=vocab_size, n_gram=n_gram_n, bits_per_token=bits, loss=loss)
model = autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5', vocab_size=vocab_size, bits_per_token=bits)
y_pred = model.predict(X_test)
y_pred = y_pred.round(0)

#################### Step 11: Encode

encode = autoencoder.encode()
if n_gram_n == 1:
    vocabulary_binary_embeddings_df = pd.read_csv(f'assets/bin_embeddings/software_requirements/no_stopwords/{n_gram_n}_gray_grams/binary_embeddings.csv', dtype=str)
    vocabulary_binary_embeddings = utils.upload_data_to_train(f'assets/bin_embeddings/software_requirements/no_stopwords/{n_gram_n}_gray_grams/binary_embeddings.csv', 'embedding')
    n_grams_embeddings = encode.predict(vocabulary_binary_embeddings)
    n_grams = vocabulary_binary_embeddings_df['lambda_gram']
else:
    n_grams_embeddings = encode.predict(X_test)
    n_grams_df = pd.read_csv(f'data/{problem}/{n_gram}_grams/X_test.csv')
    n_grams = n_grams_df['X_text']


df_embeddings = pd.DataFrame(n_grams_embeddings)
df_embeddings.to_csv(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_embedding.tsv', sep='\t', index=False, header=False)

n_grams.to_csv(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_words.tsv', sep='\t', index=False, header=False)

##################### Step 12: Decode
ind = 2
n_grams_df = pd.read_csv(f'data/{problem}/{n_gram}_grams/X_test.csv', dtype=str)

n_grams = n_grams_df['X_text']
n_grams_embeddings = np.loadtxt(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_embedding.tsv', delimiter='\t')

autoencoder = Autoencoder(X_train.shape[1], y_train.shape[1], vocab_size=vocab_size, n_gram=n_gram_n, bits_per_token=bits, loss=loss)
model = autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5', vocab_size=vocab_size, bits_per_token=bits)
decode = autoencoder.decode()

bin_embedding = decode.predict(n_grams_embeddings)
bin_embedding = np.round(bin_embedding, 0)

# print(f'Original Embedding: {n_grams_embeddings[ind]}')
original_binary_embedding = n_grams_df['X_emb'][ind]
print(f'Kari: Original Binary embedding: {original_binary_embedding}')
karina_bin_predicted = str(bin_embedding[ind]).replace('[', '').replace('.','').replace(' ','').replace(']','').replace('\n','')
print(f'Kari: Predicted Binary embedding: {karina_bin_predicted}')
print(f'Kari: Original N-grams: {n_grams[ind]}')
if code=='bcd':
    print(f'Kari: Predicted N-grams:{utils.binary_to_ngrams(bin_embedding, ind, int(n_gram_n), dictionary, bits=bits, code=code)}')
elif code=='bp':
    print(f'Kari: Predicted N-grams:{utils.binary_to_ngrams(bin_embedding, ind, int(n_gram), dictionary, code=code)}')
elif code == 'gray':
    # numpy_vect = np.array([[0.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.]])
    # print(numpy_vect)
    print(f'Kari: Predicted N-grams:{utils.binary_to_ngrams(bin_embedding, ind, int(n_gram_n), dictionary, bits=bits, code=code)}')


print(f'{n_gram}-grams: For {problem} problem {bits} bits are required')
