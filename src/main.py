# from src.model.ffnn import FFNN
# from src.view.gui import GUI
# from src.model.graph.model import GraphModel
# import numpy as np
# from PyQt6.QtWidgets import QApplication
# import sys
# from src.config.graphConfig import GraphConfig


# def main():

#     # Inisialisasi model
#     model = FFNN(
#         jumlah_neuron=[3, 2, 3, 5],
#         fungsi_aktivasi=["Sigmoid", "Sigmoid", "Sigmoid"],
#         fungsi_loss="CategoricalCrossEntropy",
#         inisialisasi_bobot="normal",
#         seed=123123,
#         lower_bound=-1,
#         upper_bound=1,
#         mean=0,
#         std=1,
#     )

#     x = np.array([[0, 0, 0], [4, 1, 2], [992, -992, 122], [3, 3, 3], [4, 4, 4]])
#     y = np.array([0, 1, 2, 3, 4])

#     # Training model
#     model.fit(X=x, y=y, batch_size=4, lr=0.1, epoch=40, verbose=1)

#     # print(model.hasil[0])

#     # Print model
#     # print(model)
    
#     pred = model.predict(x)
#     final_model = []

#     for i, data in enumerate(model.hasil[-1]):
#         final_model.append(data[i])
#     print(final_model)
#     app = QApplication(sys.argv)
#     model = GraphModel.create_from_layers(final_model)
#     ui = GUI(model)
#     sys.exit(app.exec())

#     # print(pred)


# if __name__ == "__main__":
#     main()

from src.model.ffnn import (
    FFNN,
)
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def main():
    train_samples = 5000

    # Inisialisasi model
    model = FFNN(
        jumlah_neuron=[784, 128, 64,64, 10],
        fungsi_aktivasi=["ReLU", "ReLU", "ReLU","Softmax"],
        fungsi_loss="CategoricalCrossEntropy",
        inisialisasi_bobot="xavier-normal",
        seed=123123,
        lower_bound=-1,
        upper_bound=1,
        mean=0,
        std=1,
    )
# # 
#     X = np.array([[0, 0, 0], [4, 1, 2], [992, -992, 122], [3, 3, 3], [4, 4, 4]])
#     y = np.array([0, 1, 2, 3, 4])

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=10000, random_state=123123
    )
    


    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))


    print(y_train.shape)

    # Training model
    model.fit(X=X_train, y=y_train, batch=4, lr=0.01, epoch=38, verbose=1)

    # Print model
    # print(model)

    pred = model.predict(X_test)
    print(pred)


if __name__ == "__main__":
    main()