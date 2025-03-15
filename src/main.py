from src.model.ffnn import FFNN, WeightInitiationMethod,ActivationFunctionMethod, LossFunctionMethod
import numpy as np


def main():

    
    # Inisialisasi model
    model = FFNN(
        jumlah_neuron=[3, 2, 3, 5],
        fungsi_aktivasi=["Sigmoid", "Sigmoid", "Sigmoid"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="normal",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    x = np.array([[0, 0, 0], [4, 1, 2], [992, -992, 122], [3, 3, 3], [4, 4, 4]])
    y = np.array([0, 1, 2, 3, 4])

    # Training model
    model.fit(X=x, y=y, batch=4, lr=0.1, epoch=1000, verbose=1)

    # Print model
    # print(model)

    pred = model.predict(x)
    print(pred)


if __name__ == "__main__":
    main()
