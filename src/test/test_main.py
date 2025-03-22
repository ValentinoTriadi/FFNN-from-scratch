import numpy as np
import os, sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.model.ffnn2 import FFNN2


MODEL_FILENAME = "ffnn_model.pkl"


def test_main():
    train_samples = 5000

    model = FFNN2(
        jumlah_neuron=[784, 128, 64, 64, 10],
        fungsi_aktivasi=["ReLU", "ReLU", "ReLU", "Softmax"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="xavier-normal",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    # Cek apakah model sudah ada
    if os.path.exists(MODEL_FILENAME):
        model = FFNN2.load_model_pickle(model, MODEL_FILENAME)
        train_needed = False
    else:
        train_needed = True
        # Inisialisasi model

    # Load MNIST dataset
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=10000, random_state=123123
    )

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    if train_needed:
        # Catat waktu mulai training
        start_time = time.time()

        # Training model
        model.fit(X=X_train, y=y_train, batch_size=100, lr=1, epochs=50)

        # Catat waktu selesai training
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

        # Simpan model setelah training
        model.save_model_pickle(MODEL_FILENAME)

    # Prediksi
    pred = model.predict(X_test)

    # Konversi y_test dari one-hot encoding ke label asli
    y_test_labels = np.argmax(y_test, axis=1)

    # Hitung F1-score
    model_accuracy = accuracy_score(y_test_labels, pred)
    print(f"Model Accuracy: {model_accuracy:.4f}")

    # Compare with tensor
    tensor_model = Sequential(
        [
            Input(shape=(784,)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    tensor_model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"]
    )

    tensor_model.fit(
        X_train,
        y_train,
        # batch_size=4,
        epochs=50,
        verbose=0,
        validation_data=(X_test, y_test),
    )

    tensor_model_accuracy = tensor_model.evaluate(X_test, y_test, verbose=1)[1]
    print(f"Tensor Accuracy: {tensor_model_accuracy:.4f}")

if __name__ == "__main__":
    test_main()
