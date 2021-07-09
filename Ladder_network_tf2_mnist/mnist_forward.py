import tensorflow as tf
import numpy as np
import random

from sklearn.metrics import accuracy_score
from ladder_net import get_forward_fc


inp_size = 28 * 28  # size of mnist dataset
n_classes = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(len(x_train), inp_size).astype("float32") / 255.0
x_test = x_test.reshape(len(x_test), inp_size).astype("float32") / 255.0

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)


# only select 100 training samples
idxs_annot = range(x_train.shape[0])
# random.seed(0)
idxs_annot = np.random.choice(x_train.shape[0], 10000)

x_train_labeled = x_train[idxs_annot]
y_train_labeled = y_train[idxs_annot]

model = get_forward_fc(
    layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes]
)

for epoch in range(100):
    print(f"Epoch {epoch} ")
    model.fit(x_train_labeled, y_train_labeled, epochs=1, verbose=2)
    y_test_pr = model.test_model.predict(x_test, batch_size=100)

    accuracy = accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1))
    print(f"Test accuracy : {accuracy}")