import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import math
import torch
from nn import *
from q5 import *
import cv2


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
# train for 200 iterations. PART of the CNN was hard coded (input/ouput size). Plz don't change the parameters
max_iters = 200
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
# work cited https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 3, kernel_size = 3, stride=1, padding=0)
      self.dropout = nn.Dropout2d(0.25)
      self.conv2 = nn.Conv2d(in_channels = 3,out_channels = 9, kernel_size = 3, stride=1, padding=0)
      self.dropout2 = nn.Dropout2d(0.25)
      self.fc = nn.Linear(324,36,bias = True)





    # x represents our data
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2 )
      x = self.dropout(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2 )
      x = self.dropout2(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      
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
        if xb.shape[0]!=batch_size:
            continue
        xb = xb.reshape((batch_size,32,32))
        xb = np.expand_dims(xb, axis=1)
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

    valid_x = valid_x.reshape((valid_x.shape[0],32,32))
    valid_x = np.expand_dims(valid_x, axis=1)
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
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))


    bboxes, bw = findLetters(im1)
    
    
    plt.imshow(bw,cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    lines = dict()
    totalh = 0
    count = 0
    # my strategy is clustering
    # start with finding the average heigh 
    for bbox in bboxes:
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        totalh += abs(y2-y1)
        count += 1
    avgh = totalh/count
    for bbox in bboxes:
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        center_row = (y1+y2)/2
        if len(lines) == 0:
            lines[1] = []
            lines[1].append(center_row)
            lines[1].append(bbox)
        else:
            closest_p = -1
            closest_d = 999999999
            for row in lines.keys():
                row_c = lines[row][0]
                if abs(row_c-center_row) < closest_d:
                    closest_p  = row
                    closest_d = abs(row_c-center_row)
            if closest_d <= avgh:
                lines[closest_p][0] = (lines[closest_p][0]*(len(lines[closest_p])-1)+center_row)/(len(lines[closest_p]))
                lines[closest_p].append(bbox)
            else:
                rs = len(lines)
                lines[rs+1] = []
                lines[rs+1].append(center_row)
                lines[rs+1].append(bbox)


    
    newbox = []
    for k in lines.keys():
        ar = lines[k][1:]
        # sort in a row based on y
        ar = sorted(ar,key=lambda l:l[3])
        newbox.extend(ar)
        newbox.append('newline')


    input = []
    for bbox in newbox:
        if bbox == 'newline':
            input.append('newline')
            continue

        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        word = bw[y1:y2,x1:x2]

        height = word.shape[0]
        width = word.shape[1]
        if height > width:
            diff = height-width
            left = diff//2
            right = math.ceil(diff/2)
            word = np.pad(word, ((0,0),(left,right)), mode='constant', constant_values=1)
        if height < width:
            diff = width - height
            up= diff//2
            down = math.ceil(diff/2)
            word = np.pad(word, ((up,down),(0,0)), mode='constant', constant_values=1)
        # You can check my rows by using imshow to plot my word
        edge = int(0.2*(word.shape[0]))
        word = np.pad(word, ((edge,edge),(edge,edge)), mode='constant', constant_values=1)
        word  = np.array(word, dtype='uint8')
        word = cv2.resize(word, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        input.append((np.transpose(word)).flatten())
    import string

    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    sentence = ''
    for ele in input:
        if ele == 'newline':

                print(sentence)
                sentence = ''

        else:
            valid_x = ele.reshape((1,32,32))
            valid_x = np.expand_dims(valid_x, axis=1)
            valid_x = torch.from_numpy(np.asarray(valid_x))
            val = valid_x.to(torch.float32)
            val_outputs = my_nn(val)
            yvalhat = torch.argmax(val_outputs,dim = 1)
            yhat = yvalhat.tolist()
            yhat = yhat[0]
            sentence += ' '     
            sentence += (letters[yhat])
    print(sentence)
