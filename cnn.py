import numpy as np

import layers


class ConvNet(object):
  """
  A convolutional network with the following architecture:

  conv - relu - 2x2 max pool - fc - relu - fc - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dtype = dtype

    C,H,W = input_dim
    affine_in_dim = (H-filter_size+1)*(W-filter_size+1)*num_filters//4

    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['W2'] = weight_scale * np.random.randn(affine_in_dim, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)                           
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    conv, cache1 = layers.conv_forward(X,W1)
    relu1, cache2 = layers.relu_forward(conv)
    maxp, cache3 = layers.max_pool_forward(relu1,pool_param)
    fc1, cache4 = layers.fc_forward(maxp,W2,b2)
    relu2, cache5 = layers.relu_forward(fc1)
    scores, cache6 = layers.fc_forward(relu2,W3,b3)

    if y is None:
      return scores

    loss, grads = 0, {}
    loss, dscores = layers.softmax_loss(scores,y)
    dx3, dW3, db3 = layers.fc_backward(dscores,cache6)
    dRelu2 = layers.relu_backward(dx3,cache5)
    dx2, dW2, db2 = layers.fc_backward(dRelu2,cache4)
    dmaxp = layers.max_pool_backward(dx2.reshape(maxp.shape),cache3)
    dRelu1 = layers.relu_backward(dmaxp,cache2)
    dx,dW1 = layers.conv_backward(dRelu1,cache1)
    
    grads = {'W1':dW1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3}

    return loss, grads
