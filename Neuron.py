import numpy as np
from tinyflow_utilities import softmax, softmax_backward, sigmoid, sigmoid_backward, relu, relu_backward

class Neuron:
    def __init__(self, name = ''):
        self.name = name

class Layer:
    def __init__(self, neurons = [], type=''):
        self.input_stream = None
        self.neurons = neurons
        self.weights = None
        self.bias = None
        self.Z = None
        self.type = type
        if self.activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_backward
        elif self.activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_backward
        elif self.activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = softmax_backward

    def __rshift__(self, other):
        other.input_stream = self

    def input_data(self):
        output = self.input_stream.output()
        return output

    def output(self):
        input_data = self.input_data()
        Z = self.weights.dot(input_data) + self.bias
        self.Z = Z
        A = self.activation(Z)
        return A

    def layer_size(self):
        return len(self.neurons)

    def ouput_shape(self):
        return len(self.neurons)


    def forward_pass(self):
        return self.output()

    def init(self):
        self.weights = np.random.randn(self.layer_size(), self.input_stream.layer_size())
        self.bias = np.zeros((self.layer_size(), 1))


class InputLayer(Layer):
    def __init__(self, neurons):
        Layer.__init__(self, neurons)

    def feed(self, feed_dict):
        self.data = feed_dict

    def output(self):
        output = np.array([ self.data[neuron.name] for neuron in self.neurons ])
        output = output[np.newaxis].T
        return output

class NeuralNet:
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.labels = None

    def train(self, train_data, label):
        pass

    def inputs(self, input_layer):
        self.input_layer = input_layer
        return self

    def outputs(self, output_layer):
        self.hidden_layers[-1] >> output_layer
        self.output_layer = output_layer
        return self

    def hidden(self, layer):
        if len(self.hidden_layers) == 0:
            self.input_layer >> layer
        else:
            self.hidden_layers[-1] >> layer
        self.hidden_layers.append(layer)
        return self

    def forward_pass(self, feed_dict, labels):
        self.input_layer.feed(feed_dict)
        output = self.output_layer.output()
        cost = self.output_layer.cost(output, labels)
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
hidden_layer_1 = Layer(hidden_neurons, type='relu')

output_neurons = [Neuron() for i in range(2)]
output_layer = Layer(output_neurons, type='sigmoid')

nn = NeuralNet()
nn.inputs( input_layer )
nn.hidden( hidden_layer_1 )
nn.outputs( output_layer )
nn.init()
labels = np.array([[0 , 1], [1, 0]])
data = {'x': 1, 'y': 2}
prediction = nn.predict(data)
print(prediction)
# forward_pass(prediction)
