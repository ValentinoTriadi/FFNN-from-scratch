import numpy as np
from src.utils.activationFunction import ActivationFunction, ActivationFunctionMethod
from src.utils.lossFunction import LossFunction, LossFunctionMethod
from src.utils.weightInitiation import WeightInitiation, WeightInitiationMethod

class FFNN2:
    def __init__(
        self,
        jumlah_neuron,
        fungsi_aktivasi,
        fungsi_loss,
        inisialisasi_bobot,
        lower_bound=-1.0,
        upper_bound=1.0,
        mean=0.0,
        std=1.0,
        seed=0,
    ):
        self.jumlah_neuron = jumlah_neuron
        self.jumlah_layer = len(jumlah_neuron) - 1
        self.fungsi_aktivasi = [ActivationFunction([method]).get_activation_function(method) for method in fungsi_aktivasi]


        self.fungsi_loss = LossFunction(fungsi_loss).get_lost_function()

        self.inisialisasi_bobot = WeightInitiation(inisialisasi_bobot, self.jumlah_layer, jumlah_neuron)
        self.gradients = []

        self.init_bobot()
        self.fungsi_aktivasi_str = fungsi_aktivasi

    def init_bobot(self):
        self.bobot = self.inisialisasi_bobot.init_weights()
        for i in range(self.jumlah_layer):
            print(f"Shape bobot layer {i}: {self.bobot[i].shape}") 

    def forward(self, X):
        hasil = [X]
        for i in range(self.jumlah_layer):
            X_with_bias = np.hstack([np.ones((hasil[-1].shape[0], 1)), hasil[-1]])
            Z = np.dot(X_with_bias, self.bobot[i])
            hasil.append(self.fungsi_aktivasi[i](Z))
        return hasil

    def backward(self, hasil, y):
        deltas = [None] * self.jumlah_layer
        activation_derivative = ActivationFunction(self.fungsi_aktivasi_str).get_activation_derivative(self.fungsi_aktivasi_str[-1])

        deltas[-1] = (hasil[-1] - y) * activation_derivative(hasil[-1])
        
        for i in range(self.jumlah_layer - 2, -1, -1):
            activation_derivative_i = ActivationFunction(self.fungsi_aktivasi_str).get_activation_derivative(self.fungsi_aktivasi_str[i])
            deltas[i] = (deltas[i + 1] @ self.bobot[i + 1][1:].T) * activation_derivative_i(hasil[i + 1])

        
        gradients = []
        for i in range(self.jumlah_layer):
            X_with_bias = np.hstack([np.ones((hasil[i].shape[0], 1)), hasil[i]])
            gradients.append((X_with_bias.T @ deltas[i]) / hasil[i].shape[0])
        return gradients

    def update(self, gradients, lr):
        for i in range(self.jumlah_layer):
            self.bobot[i] -= lr * gradients[i] 

    def fit(self, X, y, batch_size, lr, epochs):
        num_samples = X.shape[0]  # Jumlah total data

        for epoch in range(epochs):
            # Acak indeks dataset tiap epoch untuk meningkatkan generalisasi
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Proses dalam batch
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward & Backward per batch
                hasil = self.forward(X_batch)
                gradients = self.backward(hasil, y_batch)

                self.gradients = gradients
                # Update bobot
                self.update(gradients, lr)

            # Hitung loss setelah satu epoch
            hasil_final = self.forward(X)
            loss = self.fungsi_loss(hasil_final[-1], y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")


    def predict(self, X):
        return np.argmax(self.forward(X)[-1], axis=1)
