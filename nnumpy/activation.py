import numpy as np

class Activation:

     
    def __call__(self, x):
        raise NotImplementedError()
    
    def backward(self, x):
        raise NotImplementedError()

class Sigmoid(Activation):


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