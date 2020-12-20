from os import listdir
from os.path import isfile, join
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd


def generate_input_from_audio_file(path_to_audio_file, resample_to=8000):
    """
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file (string): path to the audio file
    :param resample_to (int): sampling rate
    :return: spectrogram corresponding to the input file
    """

    signal, sample_rate = librosa.core.load(path_to_audio_file)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    mfcc1 = librosa.feature.mfcc(
        y=signal_resampled, sr=resample_to, n_fft=200, hop_length=80
    )
    lenmfcc = int(mfcc1.shape[1])
    return mfcc1.T, lenmfcc


def generate_target_output_from_text(target_text):
    """
    Target output is an array of indices for each character in your string.
    The indices comes from a mapping that will
    be used while decoding the ctc output.
    :param target_text (string): target string
    :return: array of indices for each character in the string
    """
    gujarati_alphabet = [
        "\u0A81",
        "\u0A82",
        "\u0A83",
        "\u0A85",
        "\u0A86",
        "\u0A87",
        "\u0A88",
        "\u0A89",
        "\u0A8A",
        "\u0A8B",
        "\u0A8C",
        "\u0A8D",
        "\u0A8F",
        "\u0A90",
        "\u0A91",
        "\u0A93",
        "\u0A94",
        "\u0A95",
        "\u0A96",
        "\u0A97",
        "\u0A98",
        "\u0A99",
        "\u0A9A",
        "\u0A9B",
        "\u0A9C",
        "\u0A9D",
        "\u0A9E",
        "\u0A9F",
        "\u0AA0",
        "\u0AA1",
        "\u0AA2",
        "\u0AA3",
        "\u0AA4",
        "\u0AA5",
        "\u0AA6",
        "\u0AA7",
        "\u0AA8",
        "\u0AAA",
        "\u0AAB",
        "\u0AAC",
        "\u0AAD",
        "\u0AAE",
        "\u0AAF",
        "\u0AB0",
        "\u0AB2",
        "\u0AB3",
        "\u0AB5",
        "\u0AB6",
        "\u0AB7",
        "\u0AB8",
        "\u0AB9",
        "\u0ABC",
        "\u0ABD",
        "\u0ABE",
        "\u0ABF",
        "\u0AC0",
        "\u0AC1",
        "\u0AC2",
        "\u0AC3",
        "\u0AC4",
        "\u0AC5",
        "\u0AC7",
        "\u0AC8",
        "\u0AC9",
        "\u0ACB",
        "\u0ACC",
        "\u0ACD",
        "\u0AD0",
        "\u0AE0",
        "\u0AE1",
        "\u0AE2",
        "\u0AE3",
        "\u0AF1",
    ]
    space_token = " "
    end_token = ">"
    blank_token = "%"
    alphabet = gujarati_alphabet + [space_token, end_token, blank_token]
    char_to_index = {}
    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []
    for char in target_text:
        try:
            y.append(char_to_index[char])
        except:
            continue

    y = np.array(y)
    leny = y.shape[0]

    return y, leny


def encode_input_and_output(t_data, lsx, lsy, lenmfcc, leny, PathDataAudios):
    """
    this function will encode all the audios and their respective
    transcripts in format required for training model
    :param t_data (DataFrame): pandas dataframe containing the audio file names and their respective transcripts
    :param lsx (list): list of all encoded input audio
    :param lsy (list): list of all encoded input transcript
    :param lenmfcc (list): list of length of all encoded input audio
    :param leny (list): list of length of all encoded transcripts
    :param PathDataAudios (int): Path to folder where all audio files are stored
    """
    for d in t_data.itertuples():
        id = str(d.Id).zfill(9)
        text = str(d.Text).lower()
        AudioFileExtention = ".wav"
        path = PathDataAudios + id + AudioFileExtention
        X, lenmfcc_per = generate_input_from_audio_file(path)
        y, leny_per = generate_target_output_from_text(text)
        lsx.append(X)
        lsy.append(y)
        lenmfcc.append(lenmfcc_per)
        leny.append(leny_per)
    return lsx, lsy, lenmfcc, leny


def feature_extraction(
    train_val_split=0.8, pad_value_audio_feature=0.0, pad_value_character_index=75
):
    """
    this function will extract the features from the audio stored at the data directory (here "./Data/Train/Audios/")
    :param train_val_split (float): percent of data in training (range: 0-1)
    :param pad_value_audio_feature (float): padding value for input features (X)
    :param pad_value_character_index (int): padding value for the formatted references (Y)
    """
    lsx = []
    lsy = []
    lenmfcc = []
    leny = []
    PathDataAudios = (
        "./Data Files/Train/Audios/"  # path to the training data audios (X)
    )

    PathDataTranscripts = "./Data Files/Train/transcription.txt"  # path to the training data transcripts (Y)
    # Read number of audio files in the ./Data/Train/Audios
    total_files = [
        f for f in listdir(PathDataAudios) if isfile(join(PathDataAudios, f))
    ]

    # total number of files
    Totald = len(total_files)
    # 0 till Traind will be training data and from Traind till end it will be validation data
    Traind = int(Totald * train_val_split)
    t_data = pd.read_csv(PathDataTranscripts, sep="\t", names=["Id", "Text"])
    t_data.head()

    # generate input feature from audio files
    lsx, lsy, lenmfcc, leny = encode_input_and_output(
        t_data[:Totald], lsx, lsy, lenmfcc, leny, PathDataAudios
    )

    # lsx = lsx[: 2000 + 428]
    # lsy = lsy[: 2000 + 428]
    # lenmfcc = lenmfcc[: 2000 + 428]
    # leny = leny[: 2000 + 428]

    # Validation data
    lsx_v, lsy_v, lenmfcc_v, leny_v = (
        lsx[Traind:],
        lsy[Traind:],
        lenmfcc[Traind:],
        leny[Traind:],
    )
    # trining data
    lsx, lsy, lenmfcc, leny = (
        lsx[:Traind],
        lsy[:Traind],
        lenmfcc[:Traind],
        leny[:Traind],
    )

    """
        maxlen parameter of "tf.keras.preprocessing.sequence.pad_sequences" removes all 
        the audio feature having length greater than its given value. It helps to limit length of the feature and is
        helpful in case of memory constraint

        If you have no RAM constraint you can use dynamic padding, that pads to max feature length. Remove 'maxlen' parameter.
    """

    lsx = tf.keras.preprocessing.sequence.pad_sequences(
        lsx, padding="post", dtype="float32", value=pad_value_audio_feature, maxlen=1000
    )

    lsy = tf.keras.preprocessing.sequence.pad_sequences(
        lsy, padding="post", value=pad_value_character_index, maxlen=100
    )

    lsx_v = tf.keras.preprocessing.sequence.pad_sequences(
        lsx_v,
        padding="post",
        dtype="float32",
        value=pad_value_audio_feature,
        maxlen=1000,
    )
    lsy_v = tf.keras.preprocessing.sequence.pad_sequences(
        lsy_v, padding="post", value=pad_value_character_index, maxlen=100
    )

    # convert to tensor so it is trainable by tensorflow
    lsx = tf.convert_to_tensor(lsx)
    lsy = tf.convert_to_tensor(lsy)

    lenmfcc = tf.convert_to_tensor(lenmfcc)
    leny = tf.convert_to_tensor(leny)

    lsx_v = tf.convert_to_tensor(lsx_v)
    lsy_v = tf.convert_to_tensor(lsy_v)

    lenmfcc_v = tf.convert_to_tensor(lenmfcc_v)
    leny_v = tf.convert_to_tensor(leny_v)

    # print("Training")
    # print("X : ", lsx.shape)
    # print("Y : ", lsy.shape)
    # print("Len_X : ", lenmfcc.shape)
    # print("Len_Y : ", leny.shape)
    # print("Validation")
    # print("X_val : ", lsx_v.shape)
    # print("Y_val : ", lsy_v.shape)
    # print("Len_X_val : ", lenmfcc_v.shape)
    # print("Len_Y_val : ", leny_v.shape)

    batch_size = 8

    train_dataset = tf.data.Dataset.from_tensor_slices((lsx, lsy, lenmfcc, leny))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (lsx_v, lsy_v, lenmfcc_v, leny_v)
    )
    validation_dataset = validation_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # flush unnecessary variables
    del lsx
    del lsy
    del lenmfcc
    del leny
    del lsx_v
    del lsy_v
    del lenmfcc_v
    del leny_v
    del t_data

    return train_dataset, validation_dataset