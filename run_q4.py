import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import copy
import tkinter
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.005
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
examples, dimenstions = train_x.shape
examples, classes = train_y.shape
initialize_weights(dimenstions,hidden_size ,params,'layer1')
initial_weights = copy.deepcopy(params['Wlayer1'])
initialize_weights(hidden_size ,classes,params,'output')

train_loss = []
train_accu = []
val_loss = []
val_acc = []


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    b = 0
    for xb,yb in batches:
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss +=  loss
        total_acc += acc
        b+= 1
        delta = probs-yb
        delta = backwards(delta,params,'output',linear_deriv)
        grad_X = backwards(delta,params,'layer1',sigmoid_deriv)
        gw_out = params['grad_W' + 'output']
        gb_out = params['grad_b' + 'output']
        gw_lay = params['grad_W' + 'layer1']
        gb_lay = params['grad_b' + 'layer1']

        params['W' + 'output'] -=learning_rate*gw_out
        params['b' + 'output'] -=learning_rate*gb_out
        params['W' + 'layer1'] -=learning_rate*gw_lay
        params['b' + 'layer1'] -=learning_rate*gb_lay

        
    total_acc /= batch_num
    total_loss/=train_x.shape[0]
    train_accu.append(total_acc)
    train_loss.append(total_loss)
    val = forward(valid_x, params, "layer1")
    probs = forward(val, params, "output", softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss/=valid_x.shape[0]
    val_loss.append(valid_loss)
    val_acc.append(valid_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%

val = forward(valid_x, params, "layer1")
probs = forward(val, params, "output", softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)

test = forward(test_x, params, "layer1")
probs = forward(test, params, "output", softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)




print('Validation accuracy: ',valid_acc)
print('Test accuracy: ',test_acc)
xlabel = np.arange(0,max_iters)



plt.plot(xlabel,np.array(train_accu),label = 'train_accuracy')
plt.plot(xlabel,np.array(val_acc),label = 'validation_accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
leg = plt.legend(loc='upper right')
plt.show()





plt.plot(xlabel,np.array(train_loss),label = 'train_loss')
plt.plot(xlabel,np.array(val_loss),label = 'validation_loss')
plt.xlabel("epochs")
plt.ylabel("loss")
leg = plt.legend(loc='upper right')
plt.show()




if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q4.3
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# ImageGrid cited from https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html
first_layer_weights = params['Wlayer1']
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8),  axes_pad=0.1,)
iamges = []
for b in range (64):
    w = first_layer_weights[:,b]
    w = w.reshape((32,32))
    iamges.append(w)
for ax, im in zip(grid, iamges):
    ax.imshow(im)

plt.show()


fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8),  axes_pad=0.1,)
iamges = []
for b in range (64):
    w = initial_weights[:,b]
    w = w.reshape((32,32))
    iamges.append(w)
for ax, im in zip(grid, iamges):
    ax.imshow(im)

plt.show()



# Q4.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
test = forward(test_x, params, "layer1")
probs = forward(test, params, "output", softmax)
print(probs.shape)
for i in range (probs.shape[0]):
    r = np.argmax(probs[i,:])
    c = np.argmax(test_y[i,:])
    confusion_matrix[r,c]+=1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
