import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import plot_model
import pickle
import pandas as pd


class CustomLoss(tf.keras.losses.Loss):
  def __init__(self, vocab_size, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = tf.constant(vocab_size, dtype=tf.float32)

  def gray_to_binary(self,gray_code):
    #asegura tipo float32
    gray_bits = tf.cast(gray_code, tf.float32)

    #obtener el número de bits
    # num_bits = tf.shape(gray_code)[1]
    # Función para calcular XOR
    def xor(prev, curr):
      return prev + curr - 2 * prev * curr

    #tf scan sobre eje de bits 
    first_bit = gray_bits[:, 0:1]  # shape (batch,1)
    rest_bits = gray_bits[:, 1:]    # shape (batch, num_bits-1)
    
    rest_bits_T = tf.transpose(rest_bits, perm=[1,0])
    binary_rest = tf.scan(xor, rest_bits_T, initializer=first_bit[:,0])
    binary_rest = tf.transpose(binary_rest, perm=[1,0])
    binary = tf.concat([first_bit, binary_rest], axis=1)
    return binary 



  def binary_decimal(self, binary_pos):
    # Asegurar tipo float32 para operaciones
    binary_pos = tf.cast(binary_pos, tf.float32)
    #obtiene elnúmero de bits
    num_bits = tf.shape(binary_pos)[1]

    #crea pesos para cada bit  2^(bit-1-i)
    powers = tf.pow(2.0, tf.cast(tf.range(num_bits - 1, -1, -1), tf.float32))

    # Multiplicar cada bit por su peso y sumar por fila (batch)
    decimal_value = tf.reduce_sum(binary_pos * powers, axis=1)

    return decimal_value/self.vocab_size

  def call(self, y_true, y_pred):
    y_true_bin=self.gray_to_binary(y_true)
    y_pred_bin=self.gray_to_binary(y_pred)

    #binario poscicional 
    y_true_decimal=self.binary_decimal(y_true_bin)
    y_pred_decimal=self.binary_decimal(y_pred_bin)
    # print(y_true_decimal)
    # print(y_pred_decimal)

    #calculo de diferencia absoluta logaritmica:
    diff = tf.abs(y_true_decimal - y_pred_decimal)
    log_diff = tf.math.log(diff + 1) / tf.math.log(2.0)
    return tf.reduce_mean(log_diff)

class Autoencoder:
    def __init__(self, input_neurons=33, input_size=33, embedding_size=200,
                 optimizer='adam', metrics=['CosineSimilarity'], vocab_size=None): #tf.keras.losses.mean_squared_logarithmic_error          #'adam'
        #hyperparameters
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = CustomLoss(vocab_size=vocab_size)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=0)

        #create the nn
        self.autoencoder = tf.keras.models.Sequential()
        self.autoencoder.add(tf.keras.layers.Dense(input_neurons, input_shape=(input_size,)))
        self.autoencoder.add(tf.keras.layers.Dropout(0.3, seed=0))

        # encoder
        encoder = tf.keras.layers.Dense(
            embedding_size, 
            activation=tf.nn.sigmoid, 
            kernel_initializer=self.initializer
        )
        self.autoencoder.add(encoder)

        # decoder
        decoder = tf.keras.layers.Dense(
            input_size, 
            activation=tf.nn.sigmoid, 
            kernel_initializer=self.initializer
        )
        self.autoencoder.add(decoder)

        #save the layers indexes to encode and decode later
        self.index_last_encoder_layer = self.autoencoder.layers.index(encoder)
        self.index_decoder_layer = self.autoencoder.layers.index(decoder)

        self.autoencoder.summary()

    def save_initialize_weights(self, initialize_weights_file='assets'):
        weights = self.autoencoder.get_weights()
        with open(initialize_weights_file, "wb") as f:
            pickle.dump(weights, f)

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=256, shuffle=False):
        self.autoencoder.compile(optimizer=self.optimizer, metrics=self.metrics, loss=self.loss)
        history = self.autoencoder.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(X_val, y_val)
        )
        return history
    
    def save(self, name='model.h5'):
        self.autoencoder.save(name)

#ommit cobj for msle , custom_objects={"KariAbsLoss": KariAbsLoss}
    def load_model(self, name='model.h5', vocab_size=None):
        # self.autoencoder = load_model(name, custom_objects={"CustomLoss": lambda: CustomLoss(vocab_size=vocab_size)}) #, custom_objects={"KariAbsLoss": KariAbsLoss}
        self.autoencoder = load_model(name, custom_objects={"CustomLoss": lambda **kwargs: CustomLoss(vocab_size=vocab_size, **kwargs)})

        return self.autoencoder
   
    def predict(self, X):
        y_pred = self.autoencoder.predict(X)
        return y_pred

    def encode(self):
        self.encoder = Model(
            inputs=self.autoencoder.input, 
            outputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output
        )
        return self.encoder

    def decode(self):
        self.decoder = Model(
            inputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output, 
            outputs=self.autoencoder.get_layer(index=self.index_decoder_layer).output
        )
        return self.decoder