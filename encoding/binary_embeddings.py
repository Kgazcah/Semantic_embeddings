
import pandas as pd

class LambdaGramEmbeddings:
    def __init__(self, word_to_indx):
        self.word_to_indx = word_to_indx

    def _embed_lambda_gram(self, words):
        binaries = [self.word_to_indx[word] for word in words if word in self.word_to_indx]
        return "".join(binaries)
    
    def _embed_lambda_gram_sep(self, words):
        return [self.word_to_indx[word] for word in words if word in self.word_to_indx]

    def bcd_to_int(self, bcd_str: str) -> int:
        digits = [str(int(bcd_str[i:i+4], 2)) for i in range(0, len(bcd_str), 4)]
        return int(''.join(digits))

    def gray_to_bin_str(self, gray_str):
        bin_str = gray_str[0]  # primer bit igual
        for i in range(1, len(gray_str)):
            # XOR del bit anterior del binario con el bit actual del Gray
            prev_bit = int(bin_str[i-1])
            gray_bit = int(gray_str[i])
            bin_str += str(prev_bit ^ gray_bit)
        return bin_str

    # def gray_to_int(self, gray_str: str) -> int:
    #     #Gray to bits
    #     gray_bits = [int(bit) for bit in gray_str]
        
    #     bin_bits = [gray_bits[0]]
        
    #     #Gray to binary
    #     for i in range(1, len(gray_bits)):
    #         bin_bits.append((bin_bits[-1] + gray_bits[i]) % 2)
        
    #     #List of bits to string
    #     bin_str = ''.join(str(b) for b in bin_bits)
    #     return int(bin_str, 2)


    def binary_to_ngrams(self, binary_embedding, ngram, dictionary, code='bp', bits=None):
        if not isinstance(binary_embedding, str):
            binary_embedding = ''.join(map(str, binary_embedding.astype(int)))
        self.binary = binary_embedding
        words = []
        inverted_dict = {v: k for k, v in dictionary.items()}

        if code == 'bp':
            # Binary Pure
            n_dim = len(self.binary) // ngram
            for i in range(ngram):
                start = i * n_dim
                end = start + n_dim
                segment = self.binary[start:end]
                word = inverted_dict.get(segment, f"<UNK:{segment}>")
                words.append(word)

        elif code == 'bcd':
            # BCD
            if bits is None:
                raise ValueError("You should specify bcd length")
            for i in range(ngram):
                start = i * bits
                end = start + bits
                segment = self.binary[start:end]

                # BCD to int
                idx = self.bcd_to_int(segment)

                # Lookup in dictionary
                word = None
                for k, v in dictionary.items():
                    if v == idx:
                        word = k
                        break
                if word is None:
                    word = f"<UNK:{segment}>"
                words.append(word)

        elif code == 'gray':
            # Gray
            if bits is None:
                raise ValueError("You should specify gray code length")
            for i in range(ngram):
                start = i * bits
                end = start + bits
                segment = self.binary[start:end]

                # Gray to binary
                # idx = self.gray_to_bin_str(segment)
                

                # Lookup in dictionary
                word = None
                for k, v in dictionary.items():
                    if v == segment:
                        word = k
                        break
                if word is None:
                    word = f"<UNK:{segment}>"
                words.append(word)

        return words


    # def binary_to_ngrams(self, binary_embedding, ngram, dictionary, code='bp', bits=None):
    #     if not isinstance(binary_embedding, str):
    #         binary_embedding = ''.join(map(str, binary_embedding.astype(int)))
    #     self.binary = binary_embedding

    #     words = []
    #     inverted_dict = {v: k for k, v in dictionary.items()}

    #     if code=='bp':
    #         #binary pure
    #         n_dim = len(self.binary) // ngram
    #         for i in range(ngram):
    #             start = i * n_dim
    #             end = start + n_dim
    #             segment = self.binary[start:end]
    #             word = inverted_dict.get(segment, f"<UNK:{segment}>")
    #             words.append(word)
    #     elif code=='bcd':
    #         # BCD
    #         if bits is None:
    #             raise ValueError("You should specify bcd lenght")

    #         for i in range(ngram):
    #             start = i * bits
    #             end = start + bits
    #             segment = self.binary[start:end]

    #             # BCD to int
    #             idx = self.bcd_to_int(segment)

    #             # Looking up for the words in the dictionary
    #             word = None
    #             for k, v in dictionary.items():
    #                 if v == idx:
    #                     word = k
    #                     break
    #             if word is None:
    #                 word = f"<UNK:{segment}>"
    #             words.append(word)
    #     return words


    def get_embeddings_df(self, lambda_grams, fun=0):
        self.lambda_grams = lambda_grams
        data = []
        for lg in self.lambda_grams:
            # divide words
            words = lg[0].split() if isinstance(lg[0], str) else lg
            if fun == 0:  
                embedding = self._embed_lambda_gram(words)
            elif fun == 1:
                embedding = self._embed_lambda_gram_sep(words)
            data.append((" ".join(words), embedding))
        return pd.DataFrame(data, columns=["lambda_gram", "embedding"])
    

    def get_embeddings_df_to_classify(self, lambda_grams):
        sentences_binary = []
        for lg in lambda_grams:
            embedding = self._embed_lambda_gram(lg)
            sentences_binary.append(embedding)
        return sentences_binary
    

    
    # def binary_to_ngrams(self, binary_embedding, ngram, dictionary):
    #     if not isinstance(binary_embedding, str):
    #         binary_embedding = ''.join(map(str, binary_embedding.astype(int)))
    #     self.binary = binary_embedding
    #     n_dim = len(self.binary) // ngram
    #     words = []
    #     inverted_dict = {v: k for k, v in dictionary.items()}
    #     for i in range(ngram):
    #         start = i * n_dim
    #         end = start + n_dim
    #         segment = self.binary[start:end]
    #         #Looking up in the dictionary
    #         word = inverted_dict.get(segment, f"<UNK:{segment}>")
    #         words.append(word)
    #     return words
