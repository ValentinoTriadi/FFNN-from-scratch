from src.model.ffnn import FFNN, WeightInitiationMethod,ActivationFunctionMethod, LossFunctionMethod
import numpy as np


def main():

    
    # Inisialisasi model
    model = FFNN(
        jumlah_neuron=[3, 2, 3, 3],
        fungsi_aktivasi=[ActivationFunctionMethod.SIGMOID, ActivationFunctionMethod.SIGMOID, ActivationFunctionMethod.SIGMOID],
        fungsi_loss=LossFunctionMethod.BinaryCrossEntropy,
        inisialisasi_bobot=WeightInitiationMethod.ZERO,
    )

    x = np.array([[0, 0, 0], [1, 1, 1]])
    y = np.array([0,1,3],dtype=float)

    # Training model
    model.fit(X=x, y=y, batch=4, lr=0.1, epoch=2, verbose=1)

    # Print model
    print(model)


if __name__ == "__main__":
    main()
