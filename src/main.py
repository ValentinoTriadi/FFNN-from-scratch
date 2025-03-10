from src.model.ffnn import FFNN
import numpy as np


def main():
    # Inisialisasi model
    model = FFNN(
        jumlah_layer=3,
        jumlah_neuron=[2, 10, 4],
        fungsi_aktivasi=["Sigmoid", "Sigmoid", "Sigmoid"],
        fungsi_loss="MSE",
        inisialisasi_bobot="uniform",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    x = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    # Training model
    model.fit(X=x, y=y, batch=4, lr=0.1, epoch=2, verbose=1)

    # Print model
    print(model)


if __name__ == "__main__":
    main()
