from feature_extractor import feature_extraction
from train_model import train_model
import pickle
import dill

dill._dill._reverse_typemap["ClassType"] = type


def train():
    # call the input feature extractor given audio paths
    train_dataset, validation_dataset = feature_extraction()

    # train model
    model = train_model(train_dataset, validation_dataset)

    # save model
    currmodel = (
        "./Models/temp_model.pickle"  # to save model with other name change this
    )
    dill.dump(model, file=open(currmodel, "wb"))


if __name__ == "__main__":
    train()