import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import copy
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
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


train_loss = []
train_accu = []
val_loss = []
val_acc = []

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First fully connected layer
      self.fc1 = nn.Linear(1024,hidden_size,bias = True)

      self.fc2 = nn.Linear(hidden_size, 36,bias = True)
    def forward(self, x):
      x = self.fc1(x)
      x = F.sigmoid(x)
      x = self.fc2(x)
      # Apply softmax to x
      return x
# learn pytorch from https://stackoverflow.com/questions/68609125/why-network-with-linear-layers-cant-learn-anything
my_nn = Net()
loss_func = nn.CrossEntropyLoss()
learning_rate = 0.009
optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate, momentum=0.5)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    b = 0

    for xb,yb in batches:
        xb = torch.from_numpy(xb)
        inputs = xb.to(torch.float32)
        yb = torch.from_numpy(yb)
        labels = yb.to(torch.float32)

        # zero the parameter gradients
        

        # forward + backward + optimize
        outputs = my_nn(inputs)
        truey = torch.argmax(labels,dim=1)
        yhat = torch.argmax(outputs,dim = 1)
        matches = torch.sum(truey == yhat)
        matches = matches.to(float)
        matches/=batch_size
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.tolist()
        total_loss +=  loss
        total_acc += matches
        b+= 1
    total_acc /= batch_num
    total_loss/=batch_num
    train_accu.append(total_acc)
    train_loss.append(total_loss)

    valid_x = torch.from_numpy(np.asarray(valid_x))
    val = valid_x.to(torch.float32)
    valid_y = torch.from_numpy(np.asarray(valid_y))
    val_l = valid_y.to(torch.float32)
    val_outputs = my_nn(val)
    truevaly = torch.argmax(val_l,dim=1)
    yvalhat = torch.argmax(val_outputs,dim = 1)
    val_matches = torch.sum(truevaly == yvalhat)
    
    valid_acc = val_matches.tolist()
    valid_acc/=valid_x.shape[0]
    valid_loss = loss_func(outputs, labels)

    valid_loss = valid_loss.tolist()
    val_loss.append(valid_loss)
    val_acc.append(valid_acc)
    optimizer.zero_grad()


    

                

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))


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
