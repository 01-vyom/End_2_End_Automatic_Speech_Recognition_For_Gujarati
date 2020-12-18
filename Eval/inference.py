import tensorflow as tf
import pandas as pd
import numpy as np
import re
import librosa
import pickle
import dill
import warnings

warnings.filterwarnings("ignore")
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import ASR

dill._dill._reverse_typemap["ClassType"] = type


def generate_input_from_audio_file_test(path_to_audio_file, resample_to=8000):
    """
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file (string): path to the audio file
    :param resample_to (int): Resampling rate
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
    """
    function to test audio samples and store the otput in a pickle file.
    :param test_file_name (string): Stores the name of the file from which the audio ID will be fetched.
    :param model_name (string): Stores model name which is used to retreive the model.
    """
    # generate the output from given trained model
    test_data = pd.read_csv(
        "Data Files/Test/" + test_file_name + ".txt", sep="\t", names=["Id", "Text"]
    )
    model = dill.load(open("./Models/" + model_name + ".pickle", "rb"))
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
    references_file_name = "./Eval/refs_" + model_name + ".pickle"
    # Model Specific Hypothesis
    hypothesis_file_name = "./Eval/hyps_" + model_name + ".pickle"
    # Save references and hypothesis
    dill.dump(refs, file=open(references_file_name, "wb"))
    dill.dump(hyps, file=open(hypothesis_file_name, "wb"))


if __name__ == "__main__":
    model = "temp_model"
    test_data = "transcription"
    eval(test_data, model)