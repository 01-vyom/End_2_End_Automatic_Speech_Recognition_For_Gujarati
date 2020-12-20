import tensorflow as tf
import numpy as np


class ASR(tf.keras.Model):
    """
    Class for defining the end to end ASR model.
    This model consists of a 1D convolutional layer followed by a bidirectional LSTM
    followed by a fully connected layer applied at each timestep.
    This is a bare-bones architecture.
    Experiment with your own architectures to get a good WER
    """

    def __init__(
        self,
        filters,
        kernel_size,
        conv_stride,
        conv_border,
        n_lstm_units,
        n_dense_units,
        _out_units_,
    ):
        super(ASR, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            strides=conv_stride,
            padding=conv_border,
            activation="relu",
        )
        self.blstm_layer1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                n_lstm_units, return_sequences=True, dropout=0.35, activation="tanh"
            )
        )
        self.blstm_layer2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                n_lstm_units, return_sequences=True, dropout=0.35, activation="tanh"
            )
        )
        self.blstm_layer3 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                n_lstm_units, return_sequences=True, dropout=0.35, activation="tanh"
            )
        )
        self.dense_layer1 = tf.keras.layers.Dense(n_dense_units, activation="relu")
        self.dense_layer2 = tf.keras.layers.Dense(n_dense_units, activation="relu")
        self.dense_layer3 = tf.keras.layers.Dense(_out_units_)

    def call(self, x):
        "Calls different layers one-by-one"
        # print("Start : ",x.shape)
        x = self.conv_layer(x)
        # print("Conv1 : ",x.shape)
        x = self.blstm_layer1(x)
        # print("BiLSTM1 : ",x.shape)
        x = self.blstm_layer2(x)
        # print("BiLSTM2 : ",x.shape)
        x = self.blstm_layer3(x)
        # print("BiLSTM3 : ",x.shape)
        x = self.dense_layer1(x)
        # print("Dense1 : ",x.shape)
        x = self.dense_layer2(x)
        # print("Dense2 : ",x.shape)
        x = self.dense_layer3(x)
        # print("Dense3 : ",x.shape)
        return x