class LambdaGrams():
  """
  A class for tokenizing text into lamda grams.

  Methods:
      get_lambda_grams(text, window_size, stride):
          Generates tokens of variable lengths using different window sizes and strides.
          Returns a list of lists containing unique tokens in the order they appear.
  """
  def __init__(self, corpus):
    self.corpus = corpus

  #getting the unique tokens of the text
  def get_lambda_grams(self, window_size, stride=1):
    tokenized_text = [sentence.split() for sentence in self.corpus]
    tokens = []
    for sentence in tokenized_text:
        for i in range(0, len(sentence) - window_size + 1, stride):
            new_sentence = sentence[i:i + window_size]
            if len(new_sentence) == 1:
                tokens.append(tuple(new_sentence))
            else:
                tokens.append((' '.join(new_sentence),))
    unique_tokens = list(dict.fromkeys(tokens))
    return [list(t) for t in unique_tokens]
  

    #getting the unique tokens of the text
  def get_lambda_grams_for_classify(self, window_size, stride=1):
    tokenized_text = [sentence.split() for sentence in self.corpus]
    sentences_tokens = []
    for sentence in tokenized_text:
        s_t = []
        for i in range(0, len(sentence) - window_size + 1, stride):
            new_sentence = sentence[i:i + window_size]
            if len(new_sentence) == 1:
                s_t.append(new_sentence)
            else:
                s_t.append(new_sentence)
        sentences_tokens.append(s_t)
    return sentences_tokens