import random
import math


# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Node:
    def __init__(self, weight=None, bias=None, activate_func=sigmoid, activate_func_derivative=sigmoid_derivative):
        self.weight = weight if weight is not None else random.uniform(-1, 1)
        self.bias = bias if bias is not None else random.uniform(-1, 1)
        self.activate_func = activate_func
        self.activate_func_derivative = activate_func_derivative
        self.output = None
        self.inputs = None
        self.delta = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activate_func(sum(inputs[i] * self.weight[i] for i in range(len(inputs))) + self.bias)
        return self.output

    def backward(self, error, learning_rate):
        self.delta = error * self.activate_func_derivative(self.output)
        for i in range(len(self.weight)):
            self.weight[i] += learning_rate * self.delta * self.inputs[i]
        self.bias += learning_rate * self.delta
        return self.delta


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Node(weight=[random.uniform(-1, 1) for _ in range(input_size)]) for _ in range(hidden_size)]
        self.output_layer = [Node(weight=[random.uniform(-1, 1) for _ in range(hidden_size)], activate_func=lambda x: x,
                                  activate_func_derivative=lambda x: 1) for _ in range(output_size)]

    def forward(self, x):
        hidden_outputs = [node.forward(x) for node in self.hidden_layer]
        final_outputs = [node.forward(hidden_outputs) for node in self.output_layer]
        return final_outputs

    def backward(self, y, learning_rate):
        output_errors = [yi - yi_pred.output for yi, yi_pred in zip(y, self.output_layer)]
        hidden_errors = [node.backward(error, learning_rate) for node, error in zip(self.output_layer, output_errors)]
        for node in self.hidden_layer:
            node.backward(sum(hidden_errors), learning_rate)

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                self.forward([xi])
                self.backward([yi], learning_rate)
            if epoch % 1000 == 0:
                predictions = [self.forward([xi])[0] for xi in x]
                loss = sum((yi - yi_pred) ** 2 for yi, yi_pred in zip(y, predictions)) / len(y)
                print(f'Epoch {epoch}, Loss: {loss}')


if __name__ == '__main__':
    # 初始化数据
    x = [-2, -1, 0, 1, 2, 3]
    y = [4, 1, 0, 1, 4, 9]  # 一元二次次函数问题： y=x^2

    # 初始化和训练神经网络
    input_size = 1
    hidden_size = 20  # 隐藏层节点数
    output_size = 1
    epochs = 10000
    learning_rate = 0.01  # 使用较小的学习率

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(x, y, epochs, learning_rate)

    # 测试训练后的神经网络
    predictions = [nn.forward([xi])[0] for xi in x]
    print("Final predictions:")
    print(predictions)