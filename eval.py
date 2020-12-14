from __future__ import division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import librosa
import pickle
import dill

dill._dill._reverse_typemap["ClassType"] = type


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


def generate_input_from_audio_file_test(path_to_audio_file, resample_to=8000):
    """
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file: path to the audio file
    :param resample_to:
    :return: spectrogram corresponding to the input file
    """
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)

    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    mfcc1 = librosa.feature.mfcc(
        y=signal_resampled, sr=resample_to, n_fft=200, hop_length=80
    )

    mfcc1 = tf.convert_to_tensor(mfcc1, dtype=tf.float32)
    tp = tf.transpose(mfcc1, [1, 0])
    return tp


def eval(test_file_name, model_name):
    # generate the output from given trained model
    test_data = pd.read_csv(
        "Data Files/Test/" + test_file_name, sep="\t", names=["Id", "Text"]
    )
    model = dill.load(open(model_name + ".pickle", "rb"))
    # test_data.head()
    hyps = []
    refs = []
    for t in test_data.itertuples():
        id = str(t.Id).zfill(9)
        text = str(t.Text).lower()
        testpath = "Data Files/Test/Audios/" + id + ".wav"
        refs.append(
            re.sub(
                r"[^\u0A81\u0A82\u0A83\u0A85\u0A86\u0A87\u0A88\u0A89\u0A8A\u0A8B\u0A8C\u0A8D\u0A8F\u0A90\u0A91\u0A93\u0A94\u0A95\u0A96\u0A97\u0A98\u0A99\u0A9A\u0A9B\u0A9C\u0A9D\u0A9E\u0A9F\u0AA0\u0AA1\u0AA2\u0AA3\u0AA4\u0AA5\u0AA6\u0AA7\u0AA8\u0AAA\u0AAB\u0AAC\u0AAD\u0AAE\u0AAF\u0AB0\u0AB2\u0AB3\u0AB5\u0AB6\u0AB7\u0AB8\u0AB9\u0ABC\u0ABD\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2\u0AC3\u0AC4\u0AC5\u0AC7\u0AC8\u0AC9\u0ACB\u0ACC\u0ACD\u0AD0\u0AE0\u0AE1\u0AE2\u0AE3\u0AF1 ]",
                "",
                text,
            )
        )
        hyps.append(
            tf.nn.softmax(
                model(
                    tf.expand_dims(
                        generate_input_from_audio_file_test(testpath), axis=0
                    )
                )
            ).numpy()[0]
        )
    # Model Specific References
    references_file_name = "refs_" + model_name + ".pickle"
    # Model Specific Hypothesis
    hypothesis_file_name = "hyps_" + model_name + ".pickle"
    # Save references and hypothesis
    dill.dump(refs, file=open(references_file_name, "wb"))
    dill.dump(hyps, file=open(hypothesis_file_name, "wb"))


if __name__ == "__main__":
    model = "MODEL_NAME.pickle"
    test_data = "transcription.txt"
    eval(test_data, model)