import numpy as np

def indices(x_shape, height, width, stride=1):
    N, C, H, W = x_shape  
    HH = (H-height)//stride + 1  
    WW = (W-width)//stride + 1 

    i0 = np.repeat(np.arange(height),width)
    i0 = np.tile(i0,C)
    i1 = stride * np.repeat(np.arange(HH),WW)
    
    j0 = np.tile(np.arange(width),height*C)
    j1 = stride * np.tile(np.arange(WW),HH)

    i = i0.reshape(-1,1) + j1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    k = np.repeat(np.arange(C),height*width).reshape(-1,1)

    indice = (k,i,j)
    return indice


def im2col(x, height, width, stride=1):
    k,i,j = indices(x.shape,height,width,stride)
    col = x[:,k,i,j]
    C = x.shape[1]
    col = col.transpose(1,2,0).reshape(height*width*C,-1)
    
    return col

def col2im(col, x, height, width, stride=1):
    N, C, H, W = x.shape
    k,i,j = indices(x.shape,height,width,stride)
    
    col_r = col.reshape(C*height*width,-1,N)
    col_r = col_r.transpose(2,0,1)
    np.add.at(x,(slice(None),k,i,j),col_r)

    return x


def conv_forward(x, w):
    N, C, H, W = x.shape
    F, C, Hk, Wk = w.shape
    HH = H - Hk + 1
    WW = W - Wk + 1
    
    col_x = im2col(x,Hk,Wk)
    col_w = w.reshape(F,-1) 
    out = col_w.dot(col_x)
    out = out.reshape(F,HH,WW,N).transpose(3,0,1,2)
    cache = (x, w, col_x,col_w)
    return out, cache


def conv_backward(dout, cache):
    x,w, col_x,col_w = cache
    N,C,H,W = x.shape
    F,C,Hk,Wk = w.shape

    dout = dout.transpose(1,2,3,0).reshape(F,-1)
    dw = (dout.dot(col_x.T)).reshape(F,C,Hk,Wk)

    dcol_out = col_w.T.dot(dout)
    dx = col2im(dcol_out, x, Hk, Wk)
    return dx, dw

x = np.random.randn(3, 2, 7, 7)
w = np.random.randn(4, 2, 3, 3)
# _,_,h,w = w.shape
# col = im2col(x,h,w)
# col2im(col,x,h,w)
out,cache = conv_forward(x,w)
conv_backward(out,cache)
pool_param = {'pool_height':3,'pool_width':4,'stride':2}

N,C,H,W = x.shape
height=pool_param['pool_height']; width=pool_param['pool_width']; stride=pool_param['stride']
out_size = (N, C, (H - height)//stride +1, (W- width)//stride + 1)
out = np.zeros(out_size)

