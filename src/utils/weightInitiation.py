import numpy as np

global_inisialisasi_bobot = ["zero", "uniform", "normal"]


class WeightInitiation:
    def __init__(
        self,
        model: str,
        jumlah_layer: int,
        jumlah_neuron: list[int],
    ):
        if model not in global_inisialisasi_bobot:
            raise ValueError("Metode inisialisasi bobot tidak valid")
        self.model = model
        self.jumlah_layer = jumlah_layer
        self.jumlah_neuron = jumlah_neuron

    def init_weights(
        self, input_count: int = 1, low: float = -1.0, high: float = 1.0, seed: int = 0
    ):
        if self.model == "zero":
            return self.zero(input_count)
        elif self.model == "uniform":
            print("low", low)
            print("high", high)
            print("seed", seed)
            return self.uniform(input_count, low, high, seed)
        elif self.model == "normal":
            return self.normal(input_count)
        else:
            raise ValueError("Metode inisialisasi bobot tidak valid")

    def zero(self, input_count: int):
        return [
            np.zeros(
                (
                    input_count + 1 if i == 0 else self.jumlah_neuron[i - 1] + 1,
                    self.jumlah_neuron[i],
                )
            )
            for i in range(self.jumlah_layer)
        ]

    def uniform(self, input_count: int, low: float, high: float, seed: int):
        np.random.seed(seed)
        return [
            np.random.uniform(
                low,
                high,
                (
                    input_count + 1 if i == 0 else self.jumlah_neuron[i - 1] + 1,
                    self.jumlah_neuron[i],
                ),
            )
            for i in range(self.jumlah_layer)
        ]

    def normal(self, input_count: int):
        # TODO: Implementasi inisialisasi bobot dengan nilai random normal
        pass
