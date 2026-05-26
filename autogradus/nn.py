import math
import random

from autogradus.engine import Value


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1.0, 1.0) / math.sqrt(nin)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def softmax(values):
    """
    :param values: list[Value]
    :return: list[Value]
    """
    exps = [v.exp() for v in values]
    exps_sum = sum(exps)
    return [exp / exps_sum for exp in exps]

def cross_entropy_loss(ypreds, gt):
    probs = softmax(ypreds)
    return -probs[gt].log()
