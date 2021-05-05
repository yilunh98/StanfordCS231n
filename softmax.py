import numpy as np
import layers 


class SoftmaxClassifier:
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design.

    We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.
    The architecture should be fc - relu - fc - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}

        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)                           
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_in)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        W1, b1 = self.params['W1'],self.params['b1']
        W3, b3 = self.params['W3'],self.params['b3']
        N,d_in = X.shape

        scores = None
        f, cache1 = layers.fc_forward(X,W1,b1)        #fc
        h, cache2 = layers.relu_forward(f)            #relu
        scores, cache3 = layers.fc_forward(h,W3,b3)   #fc

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = layers.softmax_loss(scores, y)
        dx2, dW3, db3 = layers.fc_backward(dscores, cache3)
        dx1 = layers.relu_backward(dx2, cache2)
        dx, dW1, db1 = layers.fc_backward(dx1, cache1)
        
        grads = {'W1':dW1,'b1':db1,'W3':dW3,'b3':db3}

        return loss, grads
