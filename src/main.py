from src.model.ffnn import (
    FFNN,
    WeightInitiationMethod,
    ActivationFunctionMethod,
    LossFunctionMethod,
)
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    train_samples = 5000

    # Inisialisasi model
    model = FFNN(
        jumlah_neuron=[784, 2, 3, train_samples],
        fungsi_aktivasi=["Linear", "Linear", "Linear"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="xavier-uniform",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    # x = np.array([[0, 0, 0], [4, 1, 2], [992, -992, 122], [3, 3, 3], [4, 4, 4]])
    # y = np.array([0, 1, 2, 3, 4])

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=10000
    )

    print(y_train.shape)

    # Training model
    model.fit(X=X_train, y=y_train, batch=4, lr=0.1, epoch=38, verbose=1)

    # Print model
    # print(model)

    pred = model.predict(X_test)
    print(pred)


if __name__ == "__main__":
    main()
