import numpy as np
import pdb
from tinyflow_utilities import softmax, softmax_backward, sigmoid, sigmoid_backward, relu, relu_backward
learning_rate = 0.01
class Neuron:
    def __init__(self, name = ''):
        self.name = name

class Layer:
    def __init__(self, neurons = [], activation=''):
        self.input_stream = None
        self.neurons = neurons
        self.weights = None
        self.bias = None
        self.Z_cache = None
        self.A_cache = None
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_backward
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_backward
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = softmax_backward

    def __rshift__(self, other):
        other.input_stream = self
        self.output_stream = other

    def input_data(self):
        output = self.input_stream.output()
        return output

    def output(self):
        # print('layer:', self.L)
        input_data = self.input_data()
        # print('input:', input_data)
        Z = self.weights.dot(input_data) + self.bias
        # print('output Z', Z)
        A = self.activation(Z)
        # print('output A ', self.activation_name, A)
        self.Z_cache = Z
        self.A_cache = A

        return A

    def layer_size(self):
        return len(self.neurons)

    def ouput_shape(self):
        return len(self.neurons)

    def derive(self, dA):
        prev_A = self.A_cache
        derived_activation = self.activation_derivative(self.Z_cache)
        dZ = dA * self.activation_derivative(self.Z_cache)
        dW = prev_A.T.dot(dZ)
        db = dZ
        next_dA =  self.weights.T.dot(dZ)

        # pdb.set_trace()
        self.weights -= dW * learning_rate
        self.bias -= db * learning_rate
        return next_dA

    def init(self):
        self.weights = np.random.randn(self.layer_size(), self.input_stream.layer_size())
        self.bias = np.zeros((self.layer_size(), 1))

class OutputLayer(Layer):
    def __init__(self, neurons, activation):
        Layer.__init__(self, neurons, activation)
        self.L = 'L'

    def derive(self, labels):
        dA = np.sum(self.Z_cache - labels)/len(self.neurons)
        self.input_stream.derive(dA)
        return dA

class InputLayer(Layer):
    def __init__(self, neurons):
        Layer.__init__(self, neurons)
        self.L = 0

    def feed(self, feed_dict):
        self.data = feed_dict

    def output(self):
        output = np.array([ self.data[neuron.name] for neuron in self.neurons ])
        output = output[np.newaxis].T
        self.Z_cache = output
        self.A_cache = output
        return output

class NeuralNet:
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.labels = None

    def train(self, train_data, label):
        for i in range(0, 100):
            output, cost = self.forward_pass(train_data, labels)
            print('cost', cost)
            grads = self.backward_pass(train_data, labels)

    def inputs(self, input_layer):
        self.input_layer = input_layer
        return self

    def outputs(self, output_layer):
        self.hidden_layers[-1] >> output_layer
        self.output_layer = output_layer
        return self

    def hidden(self, layer):
        if len(self.hidden_layers) == 0:
            layer.L = 1
            self.input_layer >> layer
        else:
            layer.L = len(self.hidden_layers) + 1
            self.hidden_layers[-1] >> layer
        self.hidden_layers.append(layer)
        return self

    def backward_pass(self, feed_dict, labels):
        self.output_layer.derive(labels)

    def forward_pass(self, feed_dict, labels):
        self.input_layer.feed(feed_dict)
        output = self.output_layer.output()
        cost = self.output_layer.derive(labels)
        return output, cost

    def predict(self, feed_dict):
        self.input_layer.feed(feed_dict)
        return self.output_layer.output()

    def init(self):
        for layer in self.hidden_layers:
            layer.init()
        self.output_layer.init()

inputs = [Neuron(name) for name in ['x', 'y']]
input_layer = InputLayer(inputs)

hidden_neurons = [Neuron() for i in range(3)]
hidden_layer_1 = Layer(hidden_neurons, activation='relu')

output_neurons = [Neuron() for i in range(2)]
output_layer = OutputLayer(output_neurons, activation='softmax')

nn = NeuralNet()
nn.inputs( input_layer )
nn.hidden( hidden_layer_1 )
nn.outputs( output_layer )
nn.init()
labels = np.array([[0 , 1]]).T
data = {'x': 1, 'y': 2}
prediction = nn.predict(data)
# print(prediction)
nn.train(data, labels)
# forward_pass(prediction)
