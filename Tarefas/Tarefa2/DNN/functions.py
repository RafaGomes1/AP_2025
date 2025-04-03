import numpy as np
from abc import abstractmethod

# === FUNÇÃO BASE DE LOSS ===
class Function:

    @abstractmethod
    def function(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


# === BINARY CROSS ENTROPY ===
class BinaryCrossEntropy(Function):

    def function(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + ((1 - y_true) / (1 - p))


# === MEAN SQUARED ERROR ===
class MeanSquaredError(Function):

    def function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        n = y_true.size
        return (2/n) * (y_pred - y_true)


# === MÉTRICAS ===
def correct_format(y):
    corrected_y = [np.argmax(y[i]) for i in range(len(y))]
    if len(y[0]) == 1:
        corrected_y = [np.round(y[i][0]) for i in range(len(y))]
    return np.array(corrected_y)


def accuracy(y_true, y_pred):
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray):
        y_true = correct_format(y_true)

    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray):
        y_pred = correct_format(y_pred)

    return np.sum(y_pred == y_true) / len(y_true)


def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def mse_derivative(y_true, y_pred):
    return 2 * np.sum(y_true - y_pred) / len(y_true)
