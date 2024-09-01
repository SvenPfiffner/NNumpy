class NeuralNet:

    def __init__(self, layers, loss_function):

        self.layers = layers
        self.loss_function = loss_function

    def __str__(self):

        return f"NeuralNet with {len(self.layers)} layers \n" \
            + "\n".join([str(layer) + " with " + str(activation) + " activation " for layer, activation in self.layers])\
            + f"\nLoss function: {self.loss_function}"

    def forward(self, x):

        for layer, activation in self.layers:
            x = layer(x)
            x = activation(x)

        return x
    
    def backward(self, x):

        for layer, activation in reversed(self.layers):
            x = activation.backward(x)
            x = layer.backward(x)

    def update(self, lr):

        for layer, _ in self.layers:
            layer.update(lr)

    def compute_loss(self, y_pred, y_true):

        return self.loss_function.forward(y_pred, y_true)
    
    def train(self, x, y, epochs, lr):

        for epoch in range(epochs):

            # Forward pass
            y_pred = self.forward(x)

            # Compute loss
            loss = self.compute_loss(y_pred, y)

            # Backward pass
            loss_grad = self.loss_function.backward()
            self.backward(loss_grad)

            # Update weights
            self.update(lr)

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    def predict(self, x):

        return self.forward(x)