import numpy as np
from abc import ABCMeta, abstractmethod

# === REGULATOR BASE ===
class Regulator(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, *args):
        pass


# === L1 REGULARIZATION ===
class L1Reg(Regulator):
    def __init__(self, l1_val):
        super().__init__()
        self._val = l1_val

    def update(self, n, w):
        return (self._val / (2 * n)) * np.sum(np.abs(w))


# === L2 REGULARIZATION ===
class L2Reg(Regulator):
    def __init__(self, l2_val):
        super().__init__()
        self._val = l2_val

    def update(self, n, w):
        return (self._val / (2 * n)) * np.sum(w ** 2)


# === OPTIMIZER BASE ===
class Optimizer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, *args):
        pass


# === RETAINED GRADIENT OPTIMIZER ===
class RetGradient(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.90):
        super().__init__()
        self.retained_gradient = None
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient