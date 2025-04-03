from abc import ABCMeta, abstractmethod
import numpy as np
import copy

# === LAYER BASE ===
class Layer(metaclass=ABCMeta):

    def __init__(self):
        self._input_shape = None

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, error, regulator=None):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


# === ACTIVATION LAYER BASE ===
class ActivationLayer(Layer):

    def __init__(self):
        super().__init__()
        self.output = None
        self.input = None

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error, regulator=None):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0


# === RELU ===
class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input >= 0, 1, 0)


# === SIGMOID ===
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        f_x = self.activation_function(input)
        return f_x * (1 - f_x)


# === DENSE LAYER ===
class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        super().__init__()
        self.b_opt = None
        self.w_opt = None

        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer):
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, regulator=None):
        input_error = np.dot(output_error, self.weights.T)

        if regulator is not None:
            input_error += regulator.update(self.input.shape[0], self.weights)

        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self):
        return (self.n_units,)


# === DROPOUT LAYER CORRIGIDA ===
class DropOutLayer(Layer):

    def __init__(self, n_units, drop_rate):
        super().__init__()
        self.b_opt = None
        self.w_opt = None

        self.n_units = n_units
        self._drop_rate = drop_rate

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self._tmp_weights = None

    def initialize(self, optimizer):
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def _dropout(self, training):
        if not training:
            return self.weights

        m, n = self.weights.shape
        if self._drop_rate == 1:
            return np.zeros((m, n))

        mask = np.random.rand(m, n) > self._drop_rate
        return mask * self.weights / (1.0 - self._drop_rate)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self._tmp_weights = self._dropout(training)
        self.output = np.dot(self.input, self._tmp_weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, regulator=None):
        input_error = np.dot(output_error, self._tmp_weights.T)

        if regulator is not None:
            input_error += regulator.update(self.input.shape[0], self._tmp_weights)

        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self):
        return (self.n_units,)
