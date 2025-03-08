import numpy as np

global_inisialisasi_bobot = ["zero", "uniform", "normal"]


class WeightInitiation:
    def __init__(self, model):
        self.model = model

    def init_weights(self):
        if self.model.inisialisasi_bobot == "zero":
            self.zero()
        elif self.model.inisialisasi_bobot == "uniform":
            self.uniform()
        elif self.model.inisialisasi_bobot == "normal":
            self.normal()
        else:
            raise ValueError("Metode inisialisasi bobot tidak valid")

    def zero(self):
        # TODO: Implementasi inisialisasi bobot dengan nilai 0
        pass

    def uniform(self):
        # TODO: Implementasi inisialisasi bobot dengan nilai random uniform
        pass

    def normal(self):
        # TODO: Implementasi inisialisasi bobot dengan nilai random normal
        pass
