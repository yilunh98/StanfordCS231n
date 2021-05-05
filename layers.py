from builtins import range
import numpy as np
import math

def indices(x_shape, height, width, stride=1):
    N, C, H, W = x_shape  
    HH = (H-height)//stride + 1  
    WW = (W-width)//stride + 1 

    i0 = np.repeat(np.arange(height), width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(HH), WW)

    j0 = np.tile(np.arange(width), height*C)
    j1 = stride * np.tile(np.arange(WW), HH)

    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    k = np.repeat(np.arange(C), height*width).reshape(-1,1)

    indice = (k,i,j)
    return indice


def im2col(x, height, width, stride=1):

    indice = indices(x.shape,height,width,stride)
    k,i,j = indice
    col = x[:,k,i,j]
    C = x.shape[1]
    col = col.transpose(1,2,0).reshape(height*width*C,-1)
    
    return col,indice

    # N, C, H, W = x.shape  
    # HH = (H-height)//stride + 1  
    # WW = (W-width)//stride + 1 
    # col = np.zeros((N,C,height,width,HH,WW))
    
    # for h in range(height):
    #     for w in range(width):
    #         col[:,:,h,w,:,:] = x[:,:,h:(h+stride*HH):stride,w:(w+stride*WW):stride]
    
    # col = col.transpose(0,4,5,1,2,3).reshape(N*HH*WW, -1)
    # return col

def col2im(col, indice, shape, height, width, stride=1):
    
    N, C, H, W = shape
    k,i,j = indice
    col_r = col.reshape(C*height*width,-1,N)
    col_r = col_r.transpose(2,0,1)

    x_ = np.zeros(shape)
    np.add.at(x_,(slice(None),k,i,j),col_r)

    return x_

    # HH = (H-height)//stride + 1
    # WW = (W-width)//stride + 1
    # col = col.reshape(N, HH, WW, C, height, width).transpose(0,3,4,5,1,2)

    # img = np.zeros((N, C, H+stride-1, W+stride-1))
    # for h in range(height):
    #     for w in range(width):
    #         img[:,:, h:(h+stride*HH):stride, w:(w+stride*WW):stride] += col[:,:,h,w,:,:]

    # return img[:, :, 0:H, 0:W]

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    x = x.reshape(N,-1)

    out = x.dot(w)+b
    cache = (x, w, b)

    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx = (dout.dot(w.T)).reshape(x.shape)
    x = x.reshape(x.shape[0],-1)
    dw = x.T.dot(dout)
    db = np.ones(x.shape[0]).dot(dout)
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = x.copy()
    out[(x<=0)] = 0
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dout[(x<=0)] = 0
    dx = dout

    return dx

# def rotate(x):   #rotate 180°
#     newx=x.reshape(x.size)
#     newx=newx[::-1]
#     newx=newx.reshape(x.shape)
#     return newx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    N, C, H, W = x.shape
    F, C, Hk, Wk = w.shape
    HH = H - Hk + 1
    WW = W - Wk + 1
    
    col_x, indice = im2col(x,Hk,Wk)
    col_w = w.reshape(F,-1) 
    out = col_w.dot(col_x)
    out = out.reshape(F,HH,WW,N).transpose(3,0,1,2)

    # out = np.zeros((N,F,HH,WW))
    # for n in range(N):
    #   for f in range(F):
    #     for c in range(C):
    #         out[n,f] = out[n,f] + signal.convolve(x[n,c], rotate(w[f,c]), mode='valid')
    
    cache = (x, w, col_x, indice)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    x,w,col_x,indice = cache
    N,C,H,W = x.shape
    F,C,Hk,Wk = w.shape

    dout_r = dout.transpose(1,2,3,0).reshape(F,-1)
    dw = dout_r.dot(col_x.T).reshape(F,C,Hk,Wk)

    dcol_x = w.reshape(F,-1).T.dot(dout_r)
    dx = col2im(dcol_x, indice, x.shape, Hk, Wk)
    #dx = np.zeros((N,C,H,W))  #训练时不需要dx，加快训练速度
    


    # dw = np.zeros((F,C,Hk,Wk))

    # for c in range(C):
    #   for n in range(N) :
    #     for f in range(F):
    #       dx[n,c] = dx[n,c] + signal.convolve(w[f,c],dout[n,f], mode='full')
    #   for f in range(F):
    #     for n in range(N):
    #       dw[f,c] = dw[f,c] + signal.convolve(x[n,c],rotate(dout[n,f]), mode='valid')

    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here and we can assume that the dimension of
    input and stride will not cause problem here. Output size is given by
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    N, C, H, W = x.shape
    height, width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    out_height = (H-height)//stride +1
    out_width = (W-width)//stride + 1

    x_r = x.reshape(N*C, 1, H, W)
    col_x, indice = im2col(x_r,height,width,stride)
    arg_max = np.argmax(col_x, axis=0)
    out = col_x[arg_max,np.arange(col_x.shape[1])]
    out = out.reshape(out_height,out_width,N,C).transpose(2,3,0,1)


    # out = np.zeros((N, C, out_height, out_width))

    # for h in range(out_height):
    #     for w in range(out_height): 
    #         out[:,:,h,w] = np.max(x[:,:, h*stride:(h*stride+height), w*stride:(w*stride+width)],axis=(2,3))
            
    cache = (x, pool_param, col_x,arg_max,indice)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """

    x, pool_param, col_x, arg_max, indice = cache 
    N, C, H, W = x.shape
    height, width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    dout_r = dout.transpose(2,3,0,1).flatten()
    dcol_x = np.zeros(col_x.shape)
    # for i in range(arg_max.size):
    #     dx[i,arg_max[i]] = dout_l[i]
    dcol_x[arg_max,np.arange(dcol_x.shape[1])] = dout_r
    x_r = x.reshape(N*C,1,H,W)
    dx = col2im(dcol_x,indice,x_r.shape,height,width,stride)
    dx = dx.reshape(x.shape) 

    # arg_max = np.zeros_like((out_height, out_width))
    # dx = np.zeros_like(x)
    # for n in range(N):
    #   for c in range(C):
    #     for h in range(out_height):
    #         for w in range(out_height):
    #           arg_max = np.argmax(x[n,c, h*stride:(h*stride+height), w*stride:(w*stride+width)])
    #           index = np.unravel_index(arg_max,(height,width))
    #           dx[n,c,h*stride:(h*stride+height), w*stride:(w*stride+width)][index] = dout[n,c, h, w]
        
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the j-th
      class for the i-th input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the cross-entropy loss averaged over N samples.
    - dx: Gradient of the loss with respect to x
    """
    N,_ = x.shape
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x,axis=1,keepdims=True)
    p = exp_x/sum_x
    loss = np.sum(-np.log(p[np.arange(N),y]))/N
  
    indicator = np.zeros(x.shape)
    indicator[np.arange(N),y]=1     
    dx = (p-indicator) / N

    return loss, dx



