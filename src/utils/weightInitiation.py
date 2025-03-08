import numpy as np

global_inisialisasi_bobot = ["zero", "uniform", "normal"]


class WeightInitiation:
    def __init__(self, model: str, jumlah_layer: int, jumlah_neuron: list[int]):
        self.model = model
        self.jumlah_layer = jumlah_layer
        self.jumlah_neuron = jumlah_neuron

    def init_weights(self, input_count: int = 1):
        if self.model == "zero":
            return self.zero(input_count)
        elif self.model == "uniform":
            return self.uniform(input_count)
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

    def uniform(self, input_count: int):
        # TODO: Implementasi inisialisasi bobot dengan nilai random uniform
        pass

    def normal(self, input_count: int):
        # TODO: Implementasi inisialisasi bobot dengan nilai random normal
        pass
