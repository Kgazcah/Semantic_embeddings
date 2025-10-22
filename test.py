
vocab_to_indx = {'x': 5678451, 'hello': 25, 'yet': 1984, 'zero': 1985}
max_len = max(len(str(value)) for value in vocab_to_indx.values())

bits = max_len * 4

def int_to_bcd(n: int, bits: int) -> str:
    bcd_str = ''.join(format(int(d), '04b') for d in str(n))
    return bcd_str.zfill(bits)

vocab_to_bcd = {
    word: int_to_bcd(idx, bits) for word, idx in vocab_to_indx.items()
}
print(vocab_to_bcd)

def binary_to_ngrams(self, binary_embedding, ngram, dictionary):
    if not isinstance(binary_embedding, str):
        binary_embedding = ''.join(map(str, binary_embedding.astype(int)))
    self.binary = binary_embedding
    n_dim = len(self.binary) // ngram
    words = []
    inverted_dict = {v: k for k, v in dictionary.items()}
    for i in range(ngram):
        start = i * n_dim
        end = start + n_dim
        segment = self.binary[start:end]
        #Looking up in the dictionary
        word = inverted_dict.get(segment, f"<UNK:{segment}>")
        words.append(word)
    return words

