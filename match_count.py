import numpy as np
from autoencoder.nn import Autoencoder

problem = 'software_requirements'
n_gram = '5'
yuri_embeddings = np.loadtxt(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/yuri_n_gram_embedding.tsv', delimiter='\t')
kari_embeddings = np.loadtxt(f'assets/n_gram_embeddings/{problem}/{n_gram}_grams/n_gram_embedding.tsv', delimiter='\t')

autoencoder = Autoencoder()
model= autoencoder.load_model(f'assets/models/{problem}/{n_gram}_grams/model_{n_gram}.h5')
decode = autoencoder.decode()

yuri_binary = np.round(decode.predict(yuri_embeddings)).astype(int)
kari_binary = np.round(decode.predict(kari_embeddings)).astype(int)

diffs = kari_binary != yuri_binary 
false_counts = np.sum(diffs, axis=1)

equal_indices = np.where(false_counts == 0)[0] 
equal_count = len(equal_indices)
different_sum = np.sum(false_counts)

print(f"{n_gram}_grams comparison")
print(f"Equal: {equal_count}")
print(f"Average differences: {different_sum / len(kari_binary):.2f}")
print(f"Equal indexes: {equal_indices.tolist()}")