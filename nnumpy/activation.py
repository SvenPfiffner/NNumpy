import numpy as np

class Activation:
    """
    Base class for activation functions.
    Methods:
    - __call__(self, x): Apply the activation function to the input x.
    - backward(self, x): Compute the derivative of the activation function with respect to x.
    Note: This is an abstract class and should not be instantiated directly.
    """

    def __call__(self, x):
        raise NotImplementedError()
    
    def backward(self, x):
        raise NotImplementedError()

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    This activation function applies the sigmoid function to the input.
    Methods:
    - __call__(self, x): Calls the forward method.
    - __str__(self): Returns a string representation of the activation function.
    - forward(self, x): Applies the sigmoid function to the input and returns the result.
    - backward(self, x): Computes the derivative of the sigmoid function with respect to the input.
    - _sigmoid(self, x): Computes the sigmoid function.
    """

    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return "Sigmoid"
    
    def forward(self, x):
        self.output = self._sigmoid(x)
        return self.output
    
    def backward(self, x):
        return x * self.output * (1 - self.output)
    
    def _sigmoid(self, x):
         return 1 / (1 + np.exp(-x))
    
    
class ReLU(Activation):
    """
    ReLU activation function.
    Methods:
    - __call__(self, x): Calls the forward method.
    - __str__(self): Returns a string representation of the ReLU activation function.
    - forward(self, x): Computes the forward pass of the ReLU activation function.
    - backward(self, x): Computes the backward pass of the ReLU activation function.
    """

    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return "ReLU"

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, x):
        d_input = x.copy()
        d_input[self.input <= 0] = 0
        return d_input
    
class Tanh(Activation):
    """
    Tanh activation function.
    Methods:
    - __call__(self, x): Applies the forward pass of the Tanh activation function to the input.
    - __str__(self): Returns a string representation of the Tanh activation function.
    - forward(self, x): Applies the forward pass of the Tanh activation function to the input and stores the output.
    - backward(self, x): Computes the gradient of the Tanh activation function with respect to the input.
    """

    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return "Tanh"

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
        
    def backward(self, x):
        return x * (1 - self.output ** 2)
    
class Softmax(Activation):
    """
    Softmax activation function.
    Methods:
    - __call__(self, x): Calls the forward method to compute the softmax of the input array.
    - __str__(self): Returns a string representation of the Softmax activation function.
    - forward(self, x): Computes the softmax of the input array.
    - backward(self, x): Computes the derivative of the softmax function with respect to the input array.
    """
    
    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return "Softmax"
    
    def forward(self, x):
        exp_shifted = np.exp(x - x.max(axis=1, keepdims=True))
        self.output = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
        return self.output
    
    def backward(self, x):

        deriv = np.empty_like(x)

        for i, (single_output, single_d_output) in enumerate(zip(self.output, x)):
            jacobian = np.diag(single_output) - np.outer(single_output, single_output)
            deriv[i] = np.dot(jacobian, single_d_output)

        return deriv