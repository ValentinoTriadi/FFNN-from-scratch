import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Literal
from utils.activationFunction import ActivationFunction
from utils.lossFunction import LossFunction
from utils.weightInitiation import WeightInitiation, WeightInitiationMethod
from view.gui import GUI, GraphModel
from PyQt6.QtWidgets import QApplication
import time, sys, pickle


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
        ['zero', 'uniform', 'normal', 'xavier-uniform', 'xavier-normal']
    @param verbose: int
        Menampilkan log training
        0 = Tidak menampilkan apa-apa
        1 = Menampilkan progress bar

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
        fungsi_aktivasi: list[Literal["Linear", "Sigmoid", "ReLU", "Tanh", "Softmax"]],
        fungsi_loss: Literal["MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy"],
        inisialisasi_bobot: Literal[
            "zero", "uniform", "normal", "xavier-uniform", "xavier-normal"
        ],
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        verbose: int = 0,
    ):
        self.jumlah_neuron = jumlah_neuron
        self.jumlah_layer = len(jumlah_neuron) - 1

        # Inisialisasi fungsi aktivasi
        self.fungsi_aktivasi_str = fungsi_aktivasi
        self.fungsi_aktivasi = list(
            ActivationFunction(fungsi_aktivasi).get_batch_activation_function()
        )

        # Inisialisasi fungsi loss
        self.fungsi_loss_str = fungsi_loss
        self.fungsi_loss_class = LossFunction(self.fungsi_loss_str)
        self.fungsi_loss = self.fungsi_loss_class.get_lost_function()

        # Inisialisasi bobot
        # ? Bias dimasukkan ke dalam bobot indeks terakhir
        self.inisialisasi_bobot_str = inisialisasi_bobot
        self.inisialisasi_bobot = WeightInitiation(
            inisialisasi_bobot, self.jumlah_layer, jumlah_neuron
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed
        self.init_bobot()

        self.X = None
        self.y = None
        self.hasil = None
        self.loss = None
        self.verbose = verbose
        self.weight_history = []
        self.lr = None

        # Validasi input
        try:
            self.validate_input()
        except ValueError as e:
            print(f"Error: {e}")
            raise

    def init_bobot(self):
        if self.inisialisasi_bobot_str == WeightInitiationMethod.ZERO.value:
            self.bobot = self.inisialisasi_bobot.init_weights()
        elif self.inisialisasi_bobot_str == WeightInitiationMethod.UNIFORM.value:
            self.bobot = self.inisialisasi_bobot.init_weights(
                low=self.lower_bound,
                high=self.upper_bound,
                seed=self.seed,
            )
        elif self.inisialisasi_bobot_str == WeightInitiationMethod.NORMAL.value:
            self.bobot = self.inisialisasi_bobot.init_weights(
                mean=self.mean,
                std=self.std,
                seed=self.seed,
            )
        elif self.inisialisasi_bobot_str == WeightInitiationMethod.XAVIER_UNIFORM.value:
            self.bobot = self.inisialisasi_bobot.init_weights(
                seed=self.seed,
            )
        elif self.inisialisasi_bobot_str == WeightInitiationMethod.XAVIER_NORMAL.value:
            self.bobot = self.inisialisasi_bobot.init_weights(
                seed=self.seed,
            )
        else:
            raise ValueError("Metode inisialisasi bobot tidak valid")

    def validate_input(self):
        if self.jumlah_layer < 2:
            raise ValueError("Jumlah layer minimal 3")

        if len(self.fungsi_aktivasi_str) != self.jumlah_layer:
            raise ValueError(
                "Panjang list fungsi aktivasi tidak sesuai dengan jumlah layer"
            )

        for i in range(self.jumlah_layer):
            if (
                self.fungsi_aktivasi_str[i]
                not in ActivationFunction.global_fungsi_aktivasi
            ):
                raise ValueError(
                    "Fungsi aktivasi tidak valid pada layer ke-{}".format(i)
                )

        if self.fungsi_loss_str not in LossFunction.global_fungsi_loss:
            raise ValueError("Fungsi loss tidak valid")

        if (
            self.inisialisasi_bobot_str
            not in WeightInitiation.global_inisialisasi_bobot
        ):
            raise ValueError("Metode inisialisasi bobot tidak valid")

        if hasattr(self, "y") and self.y is not None:
            if self.y.shape[1] != self.jumlah_neuron[-1]:
                raise ValueError(
                    "Jumlah neuron pada layer terakhir tidak sesuai dengan y (y.shape[1] harus {})".format(
                        self.jumlah_neuron[-1]
                    )
                )
        if hasattr(self, "X") and self.X is not None:
            if self.X.shape[1] != self.jumlah_neuron[0]:
                raise ValueError(
                    "Jumlah neuron pada layer input tidak sesuai dengan X (X.shape[1] harus {})".format(
                        self.jumlah_neuron[0]
                    )
                )

        return True

    def forward(self, X: np.ndarray):
        """
        Melakukan feed forward pada model

        @param X: np.ndarray
            Input data
        @return: np.ndarray
            Output dari model
        """
        hasil = [X]
        for i in range(self.jumlah_layer):
            X_with_bias = np.clip(
                np.hstack([np.ones((hasil[-1].shape[0], 1)), hasil[-1]]), 0.000001, None
            )
            Z = np.dot(X_with_bias, self.bobot[i])
            hasil.append(self.fungsi_aktivasi[i](Z))
        return hasil

    def backward(self, hasil: np.ndarray, y: np.ndarray):
        """
        Melakukan backpropagation pada model.
        """
        deltas = [None] * self.jumlah_layer
        loss_derivative = self.fungsi_loss_class.get_loss_derivative()
        activation_derivative = ActivationFunction(
            self.fungsi_aktivasi_str
        ).get_activation_derivative(self.fungsi_aktivasi_str[-1])

        if self.fungsi_aktivasi_str[-1] == "Softmax":
            loss_grad = loss_derivative(hasil[-1], y)
            deltas[-1] = activation_derivative(hasil[-1], loss_grad)
        else:
            deltas[-1] = loss_derivative(hasil[-1], y) * activation_derivative(
                hasil[-1]
            )

        for i in range(self.jumlah_layer - 2, -1, -1):
            activation_derivative_i = ActivationFunction(
                self.fungsi_aktivasi_str
            ).get_activation_derivative(self.fungsi_aktivasi_str[i])

            upstream_gradient = deltas[i + 1] @ self.bobot[i + 1][1:].T

            if self.fungsi_aktivasi_str[i] == "Softmax":
                # Compute delta = (upstream_gradient) @ Jacobian
                deltas[i] = activation_derivative_i(
                    hasil[i + 1], grad_loss=upstream_gradient
                )
            else:
                deltas[i] = upstream_gradient * activation_derivative_i(hasil[i + 1])

        gradients = []
        for i in range(self.jumlah_layer):
            X_with_bias = np.hstack([np.ones((hasil[i].shape[0], 1)), hasil[i]])
            gradients.append((X_with_bias.T @ deltas[i]) / hasil[i].shape[0])
        return gradients

    def update(self, gradients):
        """
        Melakukan update bobot berdasarkan gradien yang dihitung
        """
        for i in range(self.jumlah_layer):
            self.bobot[i] -= self.lr * gradients[i]

    def fit(self, X: np.ndarray, y: np.ndarray, batch: int, lr: float, epochs: int):
        """
        Melatih model dengan data latih
        @param X: np.ndarray
            Data latih
        @param y: np.ndarray
            Target data latih
        @param batch: int
            Ukuran batch
        @param lr: float
            Learning rate
        @param epochs: int
            Jumlah epoch
        """
        self.X = X
        self.y = y
        self.lr = lr
        self.batch = batch

        try:
            self.validate_input()
        except ValueError as e:
            print(f"Error: {e}")
            raise

        num_samples = X.shape[0]
        now = time.time()

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            def process_batch(start_idx):
                end_idx = start_idx + batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward & Backward per batch
                hasil = self.forward(X_batch)
                gradients = self.backward(hasil, y_batch)

                return gradients

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(0, num_samples, batch):
                    gradients = executor.submit(process_batch, i).result()

                    self.gradients = gradients

                    # Update bobot
                    self.update(gradients)

            # loss per epoch
            hasil_final = self.forward(X)
            loss = self.fungsi_loss(hasil_final[-1], y)
            if self.verbose == 1:
                print(
                    f"\r[{('█' * int(50 * (epoch + 1) / epochs)).ljust(50,'░')}] - {epoch+1}/{epochs} - {(epoch+1)*100/epochs}% - {(time.time() - now):.2f} s - Loss: {loss:.6f}",
                    end="",
                    flush=True,
                )
        if self.verbose == 1:
            print(f"\nLoss: {loss:.6f}")

    def predict(self, X: np.ndarray):
        """
        Melakukan prediksi dengan model
        @param X: np.ndarray
            Data input untuk prediksi
        @return: np.ndarray
            Hasil prediksi
        """
        return np.argmax(self.forward(X)[-1], axis=1)

    def save_model_pickle(self, filename: str):
        """
        Menyimpan model ke dalam file menggunakan pickle.

        @param filename: str
            Nama file untuk menyimpan model (format .pkl)
        """
        model_data = {
            "jumlah_neuron": self.jumlah_neuron,
            "jumlah_layer": self.jumlah_layer,
            "fungsi_aktivasi_str": self.fungsi_aktivasi_str,
            "fungsi_loss_str": self.fungsi_loss_str,
            "inisialisasi_bobot_str": self.inisialisasi_bobot_str,
            "bobot": self.bobot,
            "gradients": self.gradients if hasattr(self, "gradients") else None,
            "lr": self.lr if hasattr(self, "lr") else None,
        }

        with open(filename, "wb") as file:
            pickle.dump(model_data, file)

        print(f"Model berhasil disimpan ke {filename}")

    @staticmethod
    def load_model_pickle(self, filename: str):
        """
        Memuat model dari file pickle (.pkl).

        @param filename: str
            Nama file model yang akan dimuat
        """
        with open(filename, "rb") as file:
            model_data = pickle.load(file)

        # Set ulang parameter model
        self.jumlah_neuron = model_data["jumlah_neuron"]
        self.jumlah_layer = model_data["jumlah_layer"]
        self.fungsi_aktivasi_str = model_data["fungsi_aktivasi_str"]
        self.fungsi_loss_str = model_data["fungsi_loss_str"]
        self.inisialisasi_bobot_str = model_data["inisialisasi_bobot_str"]
        self.bobot = model_data["bobot"]
        self.gradients = model_data["gradients"]
        self.lr = model_data["lr"]

        self.fungsi_aktivasi = [
            ActivationFunction([method]).get_activation_function(method)
            for method in self.fungsi_aktivasi_str
        ]
        self.fungsi_loss_class = LossFunction(self.fungsi_loss_str)
        self.fungsi_loss = self.fungsi_loss_class.get_lost_function()
        self.inisialisasi_bobot = WeightInitiation(
            self.inisialisasi_bobot_str, self.jumlah_layer, self.jumlah_neuron
        )

        print(f"Model berhasil dimuat dari {filename}")

        return self

    def tampilkan_distribusi_bobot(self, layer: list[int]):
        """
        Menampilkan histogram distribusi bobot dari layer tertentu.

        @param layer: list[int]
            List indeks layer yang ingin ditampilkan distribusi bobotnya.
        """
        plt.figure(figsize=(10, 5))
        for i, l in enumerate(layer):
            if l >= self.jumlah_layer:
                print(f"Layer {l} tidak valid")
                continue

            plt.subplot(1, len(layer), i + 1)
            plt.hist(self.bobot[l].flatten(), bins=50, color="blue", alpha=0.7)
            plt.title(f"Distribusi Bobot Layer {l}")
            plt.xlabel("Nilai Bobot")
            plt.ylabel("Frekuensi")

        plt.tight_layout()
        plt.show()

    def tampilkan_distribusi_gradient_bobot(self, layer: list[int]):
        """
        Menampilkan histogram distribusi gradient bobot dari layer tertentu.

        @param layer: list[int]
            List indeks layer yang ingin ditampilkan distribusi gradient bobotnya.
        """
        if not hasattr(self, "gradients") or self.gradients is None:
            print("Gradien belum dihitung! Jalankan training dulu.")
            return

        plt.figure(figsize=(10, 5))
        for i, l in enumerate(layer):
            if l >= self.jumlah_layer:
                print(f"Layer {l} tidak valid")
                continue

            plt.subplot(1, len(layer), i + 1)
            gradient_data = self.gradients[l]["weights"].flatten()
            plt.hist(gradient_data, bins=50, color="red", alpha=0.7)
            plt.title(f"Distribusi Gradien Layer {l}")
            plt.xlabel("Nilai Gradien")
            plt.ylabel("Frekuensi")

        plt.tight_layout()
        plt.show()

    def model_visualize(self):
        """
        Menampilkan visualisasi model
        """
        graph_model = GraphModel(self.jumlah_neuron, self.bobot, self.gradients)
        layer_distribution_input = [0, 1, 2, 3]
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        gui = GUI(graph_model, layer_distribution_input)
        return gui
        # gui.show()
