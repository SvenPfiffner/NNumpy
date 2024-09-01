class NeuralNet:

    def __init__(self, layers, loss_function):
        """
        Initializes a NeuralNet object.

        Parameters:
        layers (list): A list of Layer objects representing the layers of the neural network.
        loss_function (str): The loss function to be used for training the neural network.

        Returns:
        None
        """

        self.layers = layers
        self.loss_function = loss_function

    def __str__(self):

        return f"NeuralNet with {len(self.layers)} layers \n" \
            + "\n".join([str(layer) + " with " + str(activation) + " activation " for layer, activation in self.layers])\
            + f"\nLoss function: {self.loss_function}"

    def forward(self, x):
        """
        Performs forward propagation through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through all the layers and activations.
        """


        for layer, activation in self.layers:
            x = layer(x)
            x = activation(x)

        return x
    
    def backward(self, x):
        """
        Performs the backward pass through the neural network.

        Args:
            x: The input data.

        Returns:
            The gradients of the input data after the backward pass.
        """

        for layer, activation in reversed(self.layers):
            x = activation.backward(x)
            x = layer.backward(x)

    def update(self, lr):
        """
        Update the neural network by applying the specified learning rate.

        Parameters:
            lr (float): The learning rate to be applied.

        Returns:
            None
        """

        for layer, _ in self.layers:
            layer.update(lr)

    def compute_loss(self, y_pred, y_true):
        """
        Computes the loss between the predicted values and the true values.

        Parameters:
            y_pred (array-like): The predicted values.
            y_true (array-like): The true values.

        Returns:
            float: The computed loss.

        """

        return self.loss_function.forward(y_pred, y_true)
    
    def train(self, x, y, epochs, lr):
        """
        Trains the neural network model.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.

        Returns:
            None
        """

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
        """
        Predicts the output for a given input.

        Parameters:
        - x: The input data.

        Returns:
        - The predicted output.

        """

        return self.forward(x)