from src.model.ffnn import FFNN


def main():
    # Inisialisasi model
    model = FFNN(
        jumlah_layer=3,
        jumlah_neuron=[64, 128, 10],
        fungsi_aktivasi=["ReLU", "ReLU", "Softmax"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="uniform",
    )

    # Print docstring
    print(FFNN.__doc__)

    # Print model
    print(model.__dict__)


if __name__ == "__main__":
    main()
