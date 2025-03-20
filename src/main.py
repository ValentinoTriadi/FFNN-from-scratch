from src.model.ffnn2 import (
    FFNN2,
    WeightInitiationMethod,
    ActivationFunctionMethod,
    LossFunctionMethod,
)
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 

def main():
    train_samples = 5000

    # Inisialisasi model
    model = FFNN2(
        jumlah_neuron=[784, 128, 64, 64, 10],
        fungsi_aktivasi=["ReLU", "ReLU", "ReLU", "Softmax"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="xavier-normal",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )

    # Load MNIST dataset
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=10000, random_state=123123
    )

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    print(y_train.shape)

    # Training model
    model.fit(X=X_train, y=y_train, batch_size=100, lr=1, epochs=100)

    # Prediksi
    pred = model.predict(X_test)

    # Konversi y_test dari one-hot encoding ke label asli
    y_test_labels = np.argmax(y_test, axis=1)

    # F1-score
    f1 = f1_score(y_test_labels, pred, average="macro")
    print(f"F1-Score: {f1:.4f}")

    # Menampilkan gambar beserta label asli dan prediksi
    num_images = 10  #jumlah gambar yang ditampilin
    indices = np.random.choice(len(X_test), num_images, replace=False)  

    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray") 
        plt.axis("off")
        plt.title(f"True: {y_test_labels[idx]}\nPred: {pred[idx]}", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
