from advertorch.attacks import CoverageData
from .ensemble import EnsembleModule


class NTAAModel:
    @property
    def neuron_data(self):
        error = "Sub-classes must implement neuron_data."
        raise NotImplementedError(error)


class NTAAEnsemble(EnsembleModule, NTAAModel):
    def __init__(self, normalize, nets):
        super(NTAAEnsemble, self).__init__(normalize, nets)
        self.__neuron_data = CoverageData(nets)

    @property
    def neuron_data(self):
        return self.__neuron_data
