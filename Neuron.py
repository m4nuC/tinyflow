class Neuron:
    def __init__(self, name = ''):
        self.name = name

class Sigmoid(Neuron):
    def __init__(self):
        pass
    def forward_pass():
        pass

class Relu(Neuron):
    def __init__(self):
        pass
    def forward_pass():
        pass

class Layer:
    def __init__(self, neurons = []):
        self.input_stream = None
        self.neurons = neurons

    def __rshift__(self, other):
        other.input_stream = self

    def output(self):
        return self.input_stream.output()

    def forward_pass(self):
        return self.output()

    def pipe(next_layer):
        pass

class InputLayer(Layer):
    def __init__(self, neurons):
        Layer.__init__(self, neurons)

    def feed(self, feed_dict):
        self.data = feed_dict

    def output(self):
        return [ self.data[neuron.name] for neuron in self.neurons ]

class NeuralNet:
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []

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

    def forward_pass(self):
        self.output_layer.output()

    def predict(self, feed_dict):
        self.input_layer.feed(feed_dict)
        return self.output_layer.output()

inputs = [Neuron(name) for name in ['x', 'y']]
input_layer = InputLayer(inputs)

hidden_neurons = [Relu() for i in range(3)]
hidden_layer_1 = Layer(hidden_neurons)

output_neurons = [Sigmoid() for i in range(3)]
output_layer = Layer(output_neurons)

nn = NeuralNet()
nn.inputs( input_layer )
nn.hidden( hidden_layer_1 )
nn.outputs( output_layer )

prediction = nn.predict({'x': 1, 'y': 2})
print(prediction)
