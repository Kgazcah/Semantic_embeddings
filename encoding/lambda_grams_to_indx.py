
class LambdaGramsToIndx():
    def __init__(self, lambda_grams, vocab_to_indexes):
        self.lambda_grams = lambda_grams
        self.vocab_to_indexes = vocab_to_indexes

    def get_lambda_grams_indx(self):
        result = {}
        for lg in self.lambda_grams:
            if len(lg) == 1 and isinstance(lg[0], str):
                words = lg[0].split()
            else:
                words = lg
            key = tuple(words)
            indexes = [self.vocab_to_indexes[word] for word in words if word in self.vocab_to_indexes]
            result[key] = indexes
        return result


        
