import numpy as np

class Loss:
    """
    Base class for defining loss functions.
    Methods:
    - __call__(y_pred, y_true): Computes the loss between predicted values and true values.
    - backward(y_pred, y_true): Computes the gradient of the loss with respect to the predicted values.
    """
    
    def __call__(self, y_pred, y_true):
        raise NotImplementedError()
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError()

class MSE(Loss):
    """
    Mean Squared Error (MSE) loss function.
    This class represents the Mean Squared Error (MSE) loss function, which is commonly used in regression problems.
    It calculates the mean squared difference between the predicted values and the true values.
    Methods:
    - __call__(y_pred, y_true): Calculates the forward pass of the MSE loss function.
    - __str__(): Returns a string representation of the MSE loss function.
    - forward(y_pred, y_true): Calculates the forward pass of the MSE loss function.
    - backward(): Calculates the backward pass of the MSE loss function.
    """

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def __str__(self):
        return "MSE"
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size