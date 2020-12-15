import tensorflow as tf
import pandas as pd
import seaborn as sns

sns.set()
history = {"epoch": [], "train_loss": [], "val_loss": []}


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


def compute_ctc_loss(logits, labels, logit_length, label_length):
    """
    function to compute CTC loss.
    Note: tf.nn.ctc_loss applies log softmax to its input automatically
    :param logits: Logits from the output dense layer
    :param labels: Labels converted to array of indices
    :param logit_length: Array containing length of each input in the batch
    :param label_length: Array containing length of each label in the batch
    :return: array of ctc loss for each element in batch
    """
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        unique=None,
        blank_index=-1,
        name=None,
    )


def train_sample(x, y, optimizer, model, lenmfcc, leny):
    """
    function perform forward and backpropagation on one batch
    :param x: one batch of input
    :param y: one batch of target
    :param optimizer: optimizer
    :param model: object of the ASR class
    :return: loss from this step
    """

    with tf.GradientTape() as tape:
        logits = model(x)
        labels = y
        input_len_mfcc = list(map(int, list(lenmfcc)))
        logits_length = [
            int((float(input_len_mfcc[i]) / float(x.shape[1])) * float(logits.shape[1]))
            for i in range(len(input_len_mfcc))
        ]
        labels_length = list(map(int, list(leny)))
        loss = compute_ctc_loss(
            logits, labels, logit_length=logits_length, label_length=labels_length
        )
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def validate_sample(x, y, model, lenmfcc, leny):
    """
    function perform forward and backpropagation on one batch
    :param x: one batch of input
    :param y: one batch of target
    :param model: object of the ASR class
    :return: loss from this step
    """
    logits = model(x)
    labels = y
    input_len_mfcc = list(map(int, list(lenmfcc)))
    logits_length = [
        int((float(input_len_mfcc[i]) / float(x.shape[1])) * float(logits.shape[1]))
        for i in range(len(input_len_mfcc))
    ]
    labels_length = list(map(int, list(leny)))
    loss = compute_ctc_loss(
        logits, labels, logit_length=logits_length, label_length=labels_length
    )
    loss = tf.reduce_mean(loss)
    return loss


def train(model, optimizer, epochs, train_dataset, validation_dataset):
    """
    function to train the model for given number of epochs
    Note:
    For this example, I am passing a single batch of input to this function
    Therefore, the loop for iterating through batches is missing
    :param model: object of class ASR
    :param optimizer: optimizer
    :param X:
    :param Y:
    :param epochs:
    :return: None
    """

    for e in range(1, epochs):
        loss = 0
        for step, (x_batch_train, y_batch_train, lenmfcc, leny) in enumerate(
            train_dataset
        ):
            loss += train_sample(
                x_batch_train, y_batch_train, optimizer, model, lenmfcc, leny
            )
        loss = loss / (step + 1)
        loss_v = 0
        for step, (x_batch_validate, y_batch_validate, lenmfcc, leny) in enumerate(
            validation_dataset
        ):
            loss_v += validate_sample(
                x_batch_validate, y_batch_validate, model, lenmfcc, leny
            )
        loss_v = loss_v / (step + 1)
        history["epoch"].append(e)
        history["train_loss"].append(int(loss))
        history["val_loss"].append(int(loss_v))
        print("Epoch: {},Train Loss: {}, Val Loss: {}".format(e, loss, loss_v))


def train_model(
    train_dataset,
    validation_dataset,
    conv_filer=200,
    kernel_size=11,
    stride=2,
    conv_border="valid",
    lstm_units=200,
    dense_units=200,
    out_units=76,
    epochs=1,
):
    """
    trains the model and also plots the training data
    :param train_dataset: it is the BatchDataset containing training data
    :param validation_dataset: it is the BatchDataset containing validation data
    :param conv_filer: number of convolution filters
    :param kernel_size: sixe of the kernel
    :param stride: stride length
    :param lstm_units: number of lstm units
    :param dense_units: number of dense units
    :param out_units: number of output units
    :param epochs: number of epochs
    """
    _conv_filter_ = conv_filer
    _kernel_size_ = kernel_size
    _stride_ = stride
    _conv_border_ = conv_border
    _lstm_units_ = lstm_units
    _dense_units_ = dense_units
    _out_units_ = out_units
    _epochs_ = epochs
    optimizer = tf.keras.optimizers.Adam()

    model = ASR(
        _conv_filter_,
        _kernel_size_,
        _stride_,
        _conv_border_,
        _lstm_units_,
        _dense_units_,
        _out_units_,
    )

    train(model, optimizer, (_epochs_ + 1), train_dataset, validation_dataset)
    global history
    history = pd.DataFrame.from_dict(history)
    ax = sns.lineplot(
        x="epoch",
        y="value",
        hue="variable",
        palette=["#3498db", "Coral"],
        data=pd.melt(history, ["epoch"]),
    )

    return model
    # currmodel = "/content/drive/My Drive/Models/Arch122_micro2000_8_70_15_15.pickle"
    # dill.dump(model, file = open(currmodel, "wb"))
    # call the ctc output generator(model trainer) given input features
