import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef

from config import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="Path to the model", default="IOT_DNL/models/single_training.keras")
args = parser.parse_args()

model = tf.keras.models.load_model(args.m)
dataset = args.m.split('/')[0]

dataset_util = DATASETS[dataset]
x_train, y_train = dataset_util.load_training_data()
x_val, y_val = dataset_util.load_validation_data()
x_test, y_test = dataset_util.load_test_data()

print(f"\n\nShape of the train data: {x_train.shape}")
print(f"Shape of the validation data: {x_val.shape}")
print(f"Shape of the test data: {x_test.shape}")


for type_, x_, y_ in (
    ("train", x_train, y_train),
    ("validation", x_val, y_val),
    ("test", x_test, y_test)
):
    print(f"\n\n{type_} results")
    print(f"Number of samples: {x_.shape[0]}")
    y_pred = model.predict(x_)
    y_pred = np.argmax(y_pred, axis=1)
    print(f"Confusion matrix:\n{confusion_matrix(y_, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_, y_pred)}")
    print(f"F1 score: {f1_score(y_, y_pred, average='macro')}")
    print(f"MCC: {matthews_corrcoef(y_, y_pred)}")