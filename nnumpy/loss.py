import numpy as np

class Loss:
    
    def __call__(self, y_pred, y_true):
        raise NotImplementedError()
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError()

class MSE(Loss):

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