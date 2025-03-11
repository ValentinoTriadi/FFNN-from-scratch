from src.model.ffnn import FFNN
import numpy as np


def main():
    # Inisialisasi model
    model = FFNN(
        jumlah_neuron=[3, 2, 3, 5],
        fungsi_aktivasi=["ReLU", "ReLU", "ReLU"],
        fungsi_loss="MSE",
        inisialisasi_bobot="xavier-uniform",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    y = np.array([-1, 0, 1, 0, -1])

    # Training model
    model.fit(X=x, y=y, batch=4, lr=0.1, epoch=10, verbose=1)

    # Print model
    print(model)

    pred = model.predict(x)
    print(pred)


if __name__ == "__main__":
    main()
