import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef

from config import DATASETS, OPTIMIZERS

tf.keras.utils.set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
parser.add_argument("-o", help=f"Optimizer {list(OPTIMIZERS.keys())}", default="Adam")
parser.add_argument("-s", help="MCC score to achieve", default=0.9, type=float)
parser.add_argument("-lr", help="Learning rate", default=0.00001, type=float)
parser.add_argument("-e", help="Number of epochs", default=100, type=int)
parser.add_argument("-b", help="Batch size", default=1024, type=int)
args = parser.parse_args()

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")

if args.o not in OPTIMIZERS.keys():
    raise ValueError(f"Optimizer name must be one of {list(OPTIMIZERS.keys())}")

folder = f"{args.d}/data"
dataset_util = DATASETS[args.d]
x_train, y_train = dataset_util.load_training_data()
x_val, y_val = dataset_util.load_validation_data()

print(f"Shape of the train data: {x_train.shape}")
print(f"Shape of the validation data: {x_val.shape}")

# Create model
model = dataset_util.create_model()

# callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True
)

# compile the model
model.compile(
    optimizer=OPTIMIZERS[args.o](learning_rate=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()

history = model.fit(
    x_train, 
    y_train,
    validation_data=(x_val, y_val),
    epochs=args.e,
    batch_size=args.b,
    verbose=1,
    callbacks=[early_stopping]
)

end = time.time()

for type_, x_, y_ in (
    ("train", x_train, y_train),
    ("validation", x_val, y_val),
):
    print(f"\n\n{type_} results")
    print(f"Number of samples: {x_.shape[0]}")
    y_pred = model.predict(x_)
    y_pred = np.argmax(y_pred, axis=1)
    print(f"Confusion matrix:\n{confusion_matrix(y_, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_, y_pred)}")
    print(f"F1 score: {f1_score(y_, y_pred, average='macro')}")
    print(f"MCC: {matthews_corrcoef(y_, y_pred)}")

model.save(f"{args.d}/models/single_training.keras")

print(f"\n\nTraining time: {end - start} seconds")