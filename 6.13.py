import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import copy
import torch
import torchvision 
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# I cited data loading method from: https://www.programcreek.com/python/example/105104/torchvision.datasets.CIFAR10
# and https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
# and https://www.youtube.com/watch?v=JTZmXhJVyRE
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# initialize layers here


if __name__ == "__main__": 
    train_loss = []
    train_accu = []
    val_loss = []
    val_acc = []
# work cited https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 9, kernel_size = 3, stride=1, padding=0)
            self.dropout = nn.Dropout2d(0.25)
            self.conv2 = nn.Conv2d(in_channels = 9,out_channels = 27, kernel_size = 3, stride=1, padding=0)
            self.dropout2 = nn.Dropout2d(0.25)
            self.fc = nn.Linear(972,36,bias = True)





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
    learning_rate = 0.0075
    epochs = 50
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate, momentum=0.5)
    for e in range(epochs):
        total_loss = 0
        total_acc = 0
        batch_num = 0
        for  i, (data, labels) in enumerate(trainloader):
            if i%200 == 0:
                print(e,i)


            
            inputs = data
            # zero the parameter gradients
            

            # forward + backward + optimize
            outputs = my_nn(inputs)


            truey = labels
            yhat = torch.argmax(outputs,dim=1)
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
            batch_num+= 1
        total_acc /= batch_num
        total_loss/=batch_num
        train_accu.append(total_acc)
        train_loss.append(total_loss)
        test_loss = 0
        test_acc = 0
        test_num = 0
        for  i, (data, labels) in enumerate(testloader):

            
            inputs = data
            # zero the parameter gradients
            

            # forward + backward + optimize
            outputs = my_nn(inputs)


            truey = labels
            yhat = torch.argmax(outputs,dim=1)
            matches = torch.sum(truey == yhat)
            matches = matches.to(float)
            matches/=batch_size
            loss = loss_func(outputs, labels)
            loss = loss.tolist()
            test_loss +=  loss
            test_acc += matches
            test_num+= 1
        optimizer.zero_grad()
        test_acc /= test_num
        test_loss/=test_num
        val_loss.append(test_loss)
        val_acc.append(test_acc)




        

                    

        if e % 1 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(e,total_loss,total_acc))
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(e,test_loss,test_acc))



    xlabel = np.arange(0,epochs)
    print(val_acc)



    plt.plot(xlabel,np.array(train_accu),label = 'train_accuracy')
    plt.plot(xlabel,np.array(val_acc),label = 'test_accuracy')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    leg = plt.legend(loc='upper right')
    plt.show()



    plt.plot(xlabel,np.array(train_loss),label = 'train_loss')
    plt.plot(xlabel,np.array(val_loss),label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    leg = plt.legend(loc='upper right')
    plt.show()

