import numpy as np

class Layer:
    """
    Base class for neural network layers.
    """
    
    def __call__(self, x):
        raise NotImplementedError()
    
    def derivative(self, delta):
        raise NotImplementedError()
    
    def update(self, lr):
        raise NotImplementedError()

class LinearLayer(Layer):

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes a layer object.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include bias. Defaults to True.
        """


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
        """
        Applies the layer to the input data.

        Parameters:
        - x: Input data to be processed by the layer.

        Returns:
        - output: Output data after applying the layer.
        """


        self.input = x
        output = np.dot(x, self.weights)

        if self.bias:
            output += self.biases

        return output
    
    def __str__(self):
        return f"LinearLayer({self.in_features}, {self.out_features})"
    
    def backward(self, delta):
        """
        Performs the backward pass of the layer.

        Args:
            delta: The error gradient of the layer's output.

        Returns:
            The error gradient of the layer's input.
        """


        # Gradient with respect to weights
        self.weights_grad = np.dot(self.input.T, delta)

        # Gradient with respect to biases
        self.biases_grad = np.sum(delta, axis=0, keepdims=True)

        return np.dot(delta, self.weights.T)
    
    def update(self, lr):
        """
        Update the weights and biases of the layer using the given learning rate.

        Parameters:
        - lr (float): The learning rate used for updating the weights and biases.

        Returns:
        - None

        """

        self.weights -= lr * self.weights_grad
        if self.bias:
            self.biases -= lr * self.biases_grad.reshape(-1)
