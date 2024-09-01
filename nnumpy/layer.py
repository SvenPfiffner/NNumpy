import numpy as np

class Layer:
    
    def __call__(self, x):
        raise NotImplementedError()
    
    def derivative(self, delta):
        raise NotImplementedError()
    
    def update(self, lr):
        raise NotImplementedError()

class LinearLayer(Layer):

    def __init__(self, in_features, out_features, bias=True):


        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Initialize weights
        self.weights = np.random.randn(in_features, out_features) * 0.01

        # Initialize bias
        if bias:
            self.biases = np.zeros(out_features)
        else:
            self.biases = None

    def __call__(self, x):


        self.input = x
        output = np.dot(x, self.weights)

        if self.bias:
            output += self.biases

        return output
    
    def __str__(self):
        return f"LinearLayer({self.in_features}, {self.out_features})"
    
    def backward(self, delta):


        # Gradient with respect to weights
        self.weights_grad = np.dot(self.input.T, delta)

        # Gradient with respect to biases
        self.biases_grad = np.sum(delta, axis=0, keepdims=True)

        return np.dot(delta, self.weights.T)
    
    def update(self, lr):

        self.weights -= lr * self.weights_grad
        if self.bias:
            self.biases -= lr * self.biases_grad.reshape(-1)
