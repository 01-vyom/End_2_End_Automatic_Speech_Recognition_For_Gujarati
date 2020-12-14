from feature_extractor import feature_extraction


def train():
    # call the input feature extractor given audio paths

    train_dataset, validation_dataset = feature_extraction()

    # _conv_filter_ = 200
    # _kernel_size_ = 11
    # _stride_ = 2
    # _conv_border_ = 'valid'
    # _lstm_units_ = 200
    # _out_units_ = 76
    # _epochs_ = 40
    # optimizer = tf.keras.optimizers.Adam()

    # model = ASR(_conv_filter_, _kernel_size_, _stride_, _conv_border_, _lstm_units_, _out_units_)

    # %%time
    # history =  {"epoch":[],"train_loss":[],"val_loss":[]}
    # train(model, optimizer, (_epochs_+1),train_dataset,validation_dataset)

    # history = pd.DataFrame.from_dict(history)
    # ax = sns.lineplot(x='epoch', y='value', hue='variable',palette = ['#3498db','Coral'],data=pd.melt(history, ['epoch']))

    # currmodel = "/content/drive/My Drive/Models/Arch122_micro2000_8_70_15_15.pickle"
    # dill.dump(model, file = open(currmodel, "wb"))
    # # call the ctc output generator(model trainer) given input features


if __name__ == "__main__":
    train()