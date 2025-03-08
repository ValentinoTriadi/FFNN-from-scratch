import numpy as np
from src.utils.activationFunction import ActivationFunction
from src.utils.lossFunction import LossFunction
from src.utils.weightInitiation import WeightInitiation

global_fungsi_aktivasi = ["Linear", "Sigmoid", "ReLU", "Tanh", "Softmax"]
global_fungsi_loss = ["MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy"]
global_inisialisasi_bobot = ["zero", "uniform", "normal"]


class FFNN:
    """
    Kelas untuk menampung model Feed Forward Neural Network

    @param jumlah_layer: int
        Jumlah layer pada model
    @param jumlah_neuron: list[int]
        Jumlah neuron pada tiap layer
    @param fungsi_aktivasi: list[str]
        Fungsi aktivasi pada tiap layer
        ['Linear', 'Sigmoid', 'ReLU', 'Tanh', 'Softmax']
    @param fungsi_loss: str
        Fungsi loss yang digunakan
        ['MSE', 'BinaryCrossEntropy', 'CategoricalCrossEntropy']
    @param inisialisasi_bobot: str
        Metode inisialisasi bobot
        ['zero', 'uniform', 'normal']
    """

    def __init__(
        self,
        jumlah_layer: int,
        jumlah_neuron: list[int],
        fungsi_aktivasi: list[str],
        fungsi_loss: str,
        inisialisasi_bobot: str,
    ):
        # Validasi input
        try:
            self.validate_input(
                jumlah_layer,
                jumlah_neuron,
                fungsi_aktivasi,
                fungsi_loss,
                inisialisasi_bobot,
            )
        except ValueError as e:
            print(e)
            return

        self.jumlah_layer = jumlah_layer
        self.jumlah_neuron = jumlah_neuron
        self.fungsi_aktivasi = fungsi_aktivasi
        self.fungsi_loss = fungsi_loss
        self.inisialisasi_bobot = inisialisasi_bobot

        # Inisialisasi bobot
        # TODO: Implementasi inisialisasi bobot
        self.bobot = []

        # inisialisasi bias
        # TODO: Implementasi inisialisasi bias
        self.bias = []

        # inisialisasi fungsi aktivasi
        # TODO: Implementasi inisialisasi fungsi aktivasi
        # ? import dari class ActivationFunction
        self.fungsi_aktivasi = []

        # inisialisasi fungsi loss
        # TODO: Implementasi inisialisasi fungsi loss
        # ? import dari class LossFunction
        self.fungsi_loss = None

    def validate_input(
        self,
        jumlah_layer: int,
        jumlah_neuron: list[int],
        fungsi_aktivasi: list[str],
        fungsi_loss: str,
        inisialisasi_bobot: str,
    ):
        if jumlah_layer < 2:
            raise ValueError("Jumlah layer minimal 2")
        if len(jumlah_neuron) != jumlah_layer:
            raise ValueError(
                "Panjang list jumlah neuron tidak sesuai dengan jumlah layer"
            )
        if len(fungsi_aktivasi) != jumlah_layer:
            raise ValueError(
                "Panjang list fungsi aktivasi tidak sesuai dengan jumlah layer"
            )
        for i in range(jumlah_layer):
            if fungsi_aktivasi[i] not in global_fungsi_aktivasi:
                raise ValueError(
                    "Fungsi aktivasi tidak valid pada layer ke-{}".format(i)
                )
        if fungsi_loss not in global_fungsi_loss:
            raise ValueError("Fungsi loss tidak valid")
        if inisialisasi_bobot not in global_inisialisasi_bobot:
            raise ValueError("Metode inisialisasi bobot tidak valid")
        return True

    def forward(self, X: np.ndarray):
        """
        Melakukan feed forward pada model

        @param X: np.ndarray
            Input data
        @return: np.ndarray
            Output dari model
        """
        pass

    def backward(self, X: np.ndarray, y: np.ndarray):
        """
        Melakukan backpropagation pada model

        @param X: np.ndarray
            Input data
        @param y: np.ndarray
            Target data
        """
        pass

    def update(self, lr: float):
        """
        Melakukan update bobot dan bias pada model

        @param lr: float
            Learning rate
        """
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch: int,
        lr: float,
        epoch: int,
        verbose: int = 0,
    ):
        """
        Melakukan training model

        @param X: np.ndarray
            Input data
        @param y: np.ndarray
            Target data
        @param batch: int
            Jumlah batch
        @param lr: float
            Learning rate
        @param epoch: int
            Jumlah epoch
        @param verbose: int
            Menampilkan log training
            0 = Tidak menampilkan apa-apa
            1 = Menampilkan progress bar
        """
        pass

    def predict(self, X: np.ndarray):
        """
        Melakukan prediksi data"
        """
        pass

    def tampilkan_model(self):
        """
        Menampilkan model
        """
        print("Jumlah Layer: ", self.jumlah_layer)
        print("Jumlah Neuron: ", self.jumlah_neuron)
        print("Fungsi Aktivasi: ", self.fungsi_aktivasi)
        print("Fungsi Loss: ", self.fungsi_loss)
        print("Inisialisasi Bobot: ", self.inisialisasi_bobot)
        print("Bobot: ", self.bobot)
        print("Bias: ", self.bias)
        print("Fungsi Aktivasi: ", self.fungsi_aktivasi)
        print("Fungsi Loss: ", self.fungsi_loss)
        print()
        return

    def tampilkan_distribusi_bobot(self, layer: list[int]):
        """
        Menampilkan distribusi bobot

        @param layer: list[int]
            Layer yang ingin ditampilkan distribusi bobotnya
        """
        pass

    def tampilkan_distribusi_gradient_bobot(self, layer: list[int]):
        """
        Menampilkan distribusi gradient bobot

        @param layer: list[int]
            Layer yang ingin ditampilkan distribusi gradient bobotnya
        """
        pass

    def save_model(self, filename: str):
        """
        Menyimpan model ke dalam file

        @param filename: str
            Nama file
        """
        pass

    def load_model(self, filename: str):
        """
        Memuat model dari file

        @param filename: str
            Nama file
        """
        pass
