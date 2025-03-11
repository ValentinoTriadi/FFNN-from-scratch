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

    @param jumlah_neuron: list[int]
        Jumlah neuron pada tiap layer (termasuk input dan output)
    @param fungsi_aktivasi: list[str]
        Fungsi aktivasi pada tiap layer
        ['Linear', 'Sigmoid', 'ReLU', 'Tanh', 'Softmax']
    @param fungsi_loss: str
        Fungsi loss yang digunakan
        ['MSE', 'BinaryCrossEntropy', 'CategoricalCrossEntropy']
    @param inisialisasi_bobot: str
        Metode inisialisasi bobot
        ['zero', 'uniform', 'normal']

    Optional Param:
    -- Weight Initiation -> uniform --
    @param lower_bound: float
        Batas bawah nilai random
    @param upper_bound: float
        Batas atas nilai random
    @param seed: int
        Seed untuk reproducibility

    -- Weight Initiation -> normal --
    @param mean: float
        Mean dari distribusi normal
    @param std: float
        Standar deviasi dari distribusi normal
    @param seed: int
        Seed untuk reproducibility
    """

    def __init__(
        self,
        jumlah_neuron: list[int],
        fungsi_aktivasi: list[str],
        fungsi_loss: str,
        inisialisasi_bobot: str,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
    ):

        self.jumlah_neuron = jumlah_neuron
        self.jumlah_layer = len(jumlah_neuron) - 1
        self.fungsi_aktivasi_str = fungsi_aktivasi
        self.fungsi_loss_str = fungsi_loss
        self.inisialisasi_bobot_str = inisialisasi_bobot

        # Validasi input
        try:
            self.validate_input()
        except ValueError as e:
            print(f"\033[91m{e}\033[0m")
            exit()

        self.fungsi_aktivasi_class = ActivationFunction(fungsi_aktivasi)
        self.fungsi_loss_class = LossFunction(fungsi_loss)
        self.inisialisasi_bobot_class = WeightInitiation(
            inisialisasi_bobot, self.jumlah_layer, jumlah_neuron
        )

        # Inisialisasi bobot
        # ? Bias dimasukkan ke dalam bobot indeks terakhir
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed

        # inisialisasi fungsi aktivasi
        self.fungsi_aktivasi = list(
            self.fungsi_aktivasi_class.get_batch_activation_function()
        )

        # inisialisasi fungsi loss
        self.fungsi_loss = self.fungsi_loss_class.get_lost_function()

        print("Model berhasil diinisialisasi")

    def __str__(self):
        ret = ""
        ret += "\033[92m\nModel Feed Forward Neural Network\033[0m"
        ret += "\033[92m\nJumlah Layer: \033[0m" + str(self.jumlah_layer)
        ret += "\033[92m\nJumlah Neuron: \033[0m" + str(self.jumlah_neuron)
        ret += "\033[92m\nFungsi Aktivasi: \033[0m" + str(self.fungsi_aktivasi_str)
        ret += "\033[92m\nFungsi Loss: \033[0m" + str(self.fungsi_loss_str)
        ret += "\033[92m\nInisialisasi Bobot: \033[0m" + str(
            self.inisialisasi_bobot_str
        )
        ret += "\033[92m\nBobot: \033[0m"
        for i in range(len(self.bobot)):
            ret += "\n\033[93mEpoch " + str(i) + ":\033[0m"
            for j in range(len(self.bobot[i])):
                ret += "\n\033[94mLayer " + str(j) + ":\033[0m\n"
                ret += str(self.bobot[i][j])
        ret += "\033[92m\nHasil: \033[0m"
        for i in range(len(self.hasil)):
            ret += "\033[93m\nEpoch " + str(i) + ":\033[0m"
            for j in range(len(self.hasil[i])):
                ret += (
                    "\033[94m\nLayer " + str(j) + ":\033[0m\n" + str(self.hasil[i][j])
                )
            ret += "\n"
        ret += "\033[92m\nLoss: \033[0m" + str(self.loss)
        ret += "\033[92m\nX: \033[0m" + str(self.X)
        ret += "\033[92m\ny: \033[0m" + str(self.y)
        ret += "\033[92m\nLearning Rate: \033[0m" + str(self.lr)
        return ret

    def validate_input(self):
        if self.jumlah_layer < 2:
            raise ValueError("Jumlah layer minimal 3")
        if len(self.fungsi_aktivasi_str) != self.jumlah_layer:
            raise ValueError(
                "Panjang list fungsi aktivasi tidak sesuai dengan jumlah layer"
            )
        for i in range(self.jumlah_layer):
            if self.fungsi_aktivasi_str[i] not in global_fungsi_aktivasi:
                raise ValueError(
                    "Fungsi aktivasi tidak valid pada layer ke-{}".format(i)
                )
        if self.fungsi_loss_str not in global_fungsi_loss:
            raise ValueError("Fungsi loss tidak valid")
        if self.inisialisasi_bobot_str not in global_inisialisasi_bobot:
            raise ValueError("Metode inisialisasi bobot tidak valid")
        if hasattr(self, "y") and self.y is not None:
            if self.y.shape[0] != self.jumlah_neuron[-1]:
                raise ValueError(
                    "Jumlah neuron pada layer terakhir tidak sesuai dengan y"
                )
        if hasattr(self, "X") and self.X is not None:
            if self.X.shape[1] != self.jumlah_neuron[0]:
                raise ValueError("Jumlah neuron pada layer input tidak sesuai dengan X")

        return True

    def print_bobot(self):
        for i in range(self.jumlah_layer):
            print("Layer ke-", i)
            print(self.bobot[i])

    def forward(self):
        """
        Melakukan feed forward pada model

        @param X: np.ndarray
            Input data
        @return: np.ndarray
            Output dari model
        """
        for i in range(self.jumlah_layer):
            if i == 0:
                XWithBias = np.hstack(
                    (
                        np.ones((self.X.shape[0], 1)),  # menambahkan kolom bias
                        self.X,
                    )
                )
                self.hasil[self.current_epoch][i] = np.matmul(
                    XWithBias,
                    self.bobot[self.current_epoch][i],
                )
            else:
                XWithBias = np.hstack(
                    (
                        np.ones((self.hasil[self.current_epoch][i - 1].shape[0], 1)),
                        self.hasil[self.current_epoch][i - 1],
                    )
                )
                self.hasil[self.current_epoch][i] = np.matmul(
                    XWithBias,
                    self.bobot[self.current_epoch][i],
                )

            # Menampilkan hasil sebelum aktivasi
            print(f"Layer {i} sebelum aktivasi: {self.hasil[self.current_epoch][i]}")

            # Terapkan fungsi aktivasi
            self.hasil[self.current_epoch][i] = self.fungsi_aktivasi[i](
                self.hasil[self.current_epoch][i]
            )

            # Menampilkan hasil setelah aktivasi
            print(f"Layer {i} setelah aktivasi: {self.hasil[self.current_epoch][i]}")

        self.loss[self.current_epoch] = self.fungsi_loss(
            self.hasil[self.current_epoch][-1], self.y
        )


    def backward(self, X: np.ndarray, y: np.ndarray):
        """
        Melakukan backpropagation pada model.
        """
        self.deltas = [None] * self.jumlah_layer
        self.gradients = [None] * self.jumlah_layer

        # Hitung delta untuk output layer
        output_layer_activation_derivative = self.fungsi_aktivasi_class.get_activation_derivative(self.fungsi_aktivasi_str[-1])
        self.deltas[self.jumlah_layer - 1] = (self.hasil[self.current_epoch][self.jumlah_layer - 1] - y) * output_layer_activation_derivative(self.hasil[self.current_epoch][-1])

        # Hitung delta untuk hidden layer (dari belakang ke depan)
        for i in range(self.jumlah_layer - 1, 0, -1):  # Dari layer output ke hidden pertama
            activation_derivative = self.fungsi_aktivasi_class.get_activation_derivative(self.fungsi_aktivasi_str[i - 1])
            self.deltas[i - 1] = np.dot(self.deltas[i], self.bobot[self.current_epoch][i][1:].T) * activation_derivative(self.hasil[self.current_epoch][i - 1])

        # Hitung gradien untuk setiap layer
        for i in range(self.jumlah_layer):
            if i == 0:
                input_to_layer = X  # Input ke layer pertama adalah X
            else:
                input_to_layer = self.hasil[self.current_epoch][i - 1]

            # Tambahkan kolom bias
            input_with_bias = np.hstack((np.ones((input_to_layer.shape[0], 1)), input_to_layer))

            # Hitung gradien bobot
            weight_grad = np.dot(input_with_bias.T, self.deltas[i]) / input_to_layer.shape[0]

            # Pisahkan bias dan bobot
            self.gradients[i] = {
                "weights": weight_grad[1:],  # Semua kecuali baris pertama (bias)
                "bias": weight_grad[0:1],  # Hanya baris pertama (bias)
            }

            print(f"Layer {i} delta shape: {self.deltas[i].shape}")
            print(f"Layer {i} weight gradient shape: {self.gradients[i]['weights'].shape}")
            print(f"Layer {i} bias gradient shape: {self.gradients[i]['bias'].shape}")



    def update(self, lr: float):
        """
        Melakukan update bobot dan bias pada model dengan cara yang benar.
        """
        for i in range(self.jumlah_layer):
            if self.gradients[i] is not None:
                print(f"Updating weights for layer {i}: previous weights:\n{self.bobot[self.current_epoch][i]}")

                # Update seluruh bobot sekaligus, termasuk bias
                self.bobot[self.current_epoch][i] -= lr * np.vstack(
                    (self.gradients[i]["bias"], self.gradients[i]["weights"])
                )

                print(f"Updated weights for layer {i}:\n{self.bobot[self.current_epoch][i]}")

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
        # Init Model Fit
        self.hasil = [np.empty(self.jumlah_layer, dtype=object) for i in range(epoch)]
        self.loss = np.zeros(epoch)
        self.X = X
        self.y = y
        self.lr = lr

        try:
            self.validate_input()
        except ValueError as e:
            print(f"\033[91m{e}\033[0m")
            exit()

        # Init Bobot
        if self.inisialisasi_bobot_str == "zero":
            self.bobot = self.inisialisasi_bobot_class.init_weights(
                self.jumlah_neuron, epoch, self.lower_bound, self.upper_bound, self.mean, self.std, self.seed
            )
        elif self.inisialisasi_bobot_str == "uniform":
            self.bobot = self.inisialisasi_bobot_class.init_weights(
                input_count=self.jumlah_neuron[0],
                low=self.lower_bound,
                high=self.upper_bound,
                seed=self.seed,
                epoch=epoch,
            )
        elif self.inisialisasi_bobot_str == "normal":
            self.bobot = self.inisialisasi_bobot_class.init_weights(
                input_count=self.jumlah_neuron[0],
                mean=self.mean,
                std=self.std,
                seed=self.seed,
                epoch=epoch,
            )
        else:
            raise ValueError("Metode inisialisasi bobot tidak valid")

        for i in range(epoch):
            self.current_epoch = i
            if verbose == 1:
                print("\033[92mEpoch ke-", i, "\033[0m")
            self.forward()
            print("masuk backward")
            self.backward(X, y)
            self.update(lr)
            if i < epoch - 1:
                self.bobot[i + 1] = [layer.copy() for layer in self.bobot[i]]
            if verbose == 1:
                print("Loss: ", self.loss[i])

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
