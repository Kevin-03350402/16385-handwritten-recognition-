
import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 3.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do  + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):

    denominator = np.sqrt(in_size+out_size)
    lower = -np.sqrt(6)/denominator
    upper = np.sqrt(6)/denominator
    W = np.random.uniform(lower, upper, size=(in_size, out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# Q 3.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    res = 1/(1+np.exp(-x))
    return res

# Q 3.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    pre_act = np.matmul(X,W)+b
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 3.2.2
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros((x.shape[0],x.shape[1]))
    for i in range (x.shape[0]):
        example = x[i,:]
        c = -np.max(example)
        x_c = example +c
        sum_exp = np.sum(np.exp(x_c))
        res[i,:] = np.exp(x_c)/sum_exp

    return res


# Q 3.2.3
# compute average loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    D = (y.shape[0])*(y.shape[1])
    loss = 0
    currect = 0
    examples = y.shape[0]
    for i in range (y.shape[0]):
        y_lab = y[i,:]
        y_hat = probs[i,:]
        if np.argmax(y_hat) == np.argmax(y_lab):
            currect+=1
        logyh  = np.log(y_hat)
        loss += np.sum(-(np.multiply(y_lab ,logyh)))

    acc = currect/examples
    
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 3.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """

    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta = delta*activation_deriv(post_act)
    grad_X = np.matmul(delta,np.transpose(W))
    grad_W = np.matmul(X.T,delta)
    grad_b =np.sum(delta,axis = 0)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 3.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    N = len(y)
    ordering = np.random.permutation(N)
    x =  x[ordering]
    y=  y[ordering]
    for i in range (0,len(x),batch_size):
        if i+batch_size < len(x):
            batches.append((x[i:i+batch_size],y[i:i+batch_size]))
        else:
            batches.append((x[i:],y[i:]))


    return batches
