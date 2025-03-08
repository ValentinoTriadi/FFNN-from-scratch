from src.model.ffnn import FFNN
import numpy as np


def main():
    # Inisialisasi model
    model = FFNN(
        jumlah_layer=3,
        jumlah_neuron=[2, 3, 4],
        fungsi_aktivasi=["Linear", "Linear", "Linear"],
        fungsi_loss="MSE",
        inisialisasi_bobot="zero",
    )

    x = np.array([[0], [0], [1], [1]])
    y = np.array([-1, 1, 1, -1])

    # Training model
    model.fit(X=x, y=y, batch=4, lr=0.1, epoch=1, verbose=1)

    # Print model
    print(model)


if __name__ == "__main__":
    main()
