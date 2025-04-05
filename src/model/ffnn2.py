import pickle
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from utils.activationFunction import ActivationFunction
from utils.lossFunction import LossFunction
from utils.weightInitiation import WeightInitiation
import time


class FFNN2:
    def __init__(
        self,
        jumlah_neuron,
        fungsi_aktivasi,
        fungsi_loss,
        inisialisasi_bobot,
        verbose=0,
        lower_bound=-1.0,
        upper_bound=1.0,
        mean=0.0,
        std=1.0,
        seed=0,
    ):
        self.verbose = verbose
        self.jumlah_neuron = jumlah_neuron
        self.jumlah_layer = len(jumlah_neuron) - 1
        self.fungsi_aktivasi = [
            ActivationFunction([method]).get_activation_function(method)
            for method in fungsi_aktivasi
        ]

        self.fungsi_loss_class = LossFunction(fungsi_loss)
        self.fungsi_loss = self.fungsi_loss_class.get_lost_function()
        self.fungsi_loss_str = fungsi_loss

        self.inisialisasi_bobot_str = inisialisasi_bobot
        self.inisialisasi_bobot = WeightInitiation(
            inisialisasi_bobot, self.jumlah_layer, jumlah_neuron
        )
        self.training_loss_results = []
        self.validation_loss_results = []

        self.init_bobot()
        self.fungsi_aktivasi_str = fungsi_aktivasi

    def init_bobot(self):
        self.bobot = self.inisialisasi_bobot.init_weights()
        # for i in range(self.jumlah_layer):
        # print(f"Shape bobot layer {i}: {self.bobot[i].shape}")

    def forward(self, X):
        hasil = [X]
        for i in range(self.jumlah_layer):
            X_with_bias = np.hstack([np.ones((hasil[-1].shape[0], 1)), hasil[-1]])
            Z = np.dot(X_with_bias, self.bobot[i])
            hasil.append(self.fungsi_aktivasi[i](Z))
        return hasil

    def backward(self, hasil, y):
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
                deltas[i] = activation_derivative_i(hasil[i + 1], grad_loss=upstream_gradient)
            else:
                deltas[i] = upstream_gradient * activation_derivative_i(hasil[i + 1])


        gradients = []
        for i in range(self.jumlah_layer):
            X_with_bias = np.hstack([np.ones((hasil[i].shape[0], 1)), hasil[i]])
            gradients.append((X_with_bias.T @ deltas[i]) / hasil[i].shape[0])
        return gradients

    def update(self, gradients, lr):
        for i in range(self.jumlah_layer):
            self.bobot[i] -= lr * gradients[i]

    def fit(self, X, y, batch, lr, epochs, X_val=None, y_val=None):
        num_samples = X.shape[0]
        now = time.time()
        for epoch in range(epochs):
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
                    self.update(gradients, lr)

            # training loss per epoch
            hasil_final = self.forward(X)
            loss = self.fungsi_loss(hasil_final[-1], y)
            self.training_loss_results.append(loss)

            # validation loss per epoch
            if X_val is not None and y_val is not None:
                hasil_val = self.forward(X_val)
                val_loss = self.fungsi_loss(hasil_val[-1], y_val)
                self.validation_loss_results.append(val_loss)
            else:
                val_loss = None
                
            if self.verbose == 1:
                val_str = f" - Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(
                    f"\r[{('█' * int(50 * (epoch + 1) / epochs)).ljust(50,'░')}] "
                    f"- {epoch+1}/{epochs} - {(epoch+1)*100/epochs:.1f}% "
                    f"- {(time.time() - now):.2f}s - Training Loss: {loss:.6f}{val_str}",
                    end="", flush=True
                )

        if self.verbose == 1:
            print(f"\nFinal Training Loss: {loss:.6f}")
            if X_val is not None:
                print(f"Final Validation Loss: {self.validation_loss_results[-1]:.6f}")

    def predict(self, X):
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
