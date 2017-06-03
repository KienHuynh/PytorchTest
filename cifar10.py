import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import cPickle

import pdb
import matplotlib.pyplot as plt

def create_one_hot(target_vector, num_class, dtype=np.float32):
    """create_one_hot
    Generate one-hot 4D tensor from a target vector of length N (num sample)
    The one-hot tensor will have the shape of (N x 1 x 1 x num_class)

    :param target_vector: Index vector, values are ranged from 0 to num_class-1

    :param num_class: number of classes/labels
    :return: target vector as a 4D tensor
    """
    one_hot = np.eye(num_class+1, num_class, dtype=dtype)
    one_hot = one_hot[target_vector]
    result = np.reshape(one_hot, (target_vector.shape[0], num_class))
    return result


# Prepare the data
def data_prepare(data_path='../data/cifar-10-batches-py/'):
    train_X = np.zeros((50000, 32*32*3), dtype=np.float32)
    train_Y = np.zeros((50000, ), dtype=np.int8)
    test_X = np.zeros((10000, 32*32*3), dtype=np.float32)
    test_Y = np.zeros((10000, ), dtype=np.int8)
    for i in range(1,6):
        file_name = data_path + ('data_batch_%d' % i)
        with open(file_name, 'rb') as f:
            batch = cPickle.load(f)
            train_X[(i-1)*10000:i*10000, :] = batch['data']
            train_Y[(i-1)*10000:i*10000] = batch['labels']

    file_name = data_path + ('test_batch')
    with open(file_name, 'rb') as f:
        batch = cPickle.load(f)
        test_X = batch['data']
        test_Y[0:10000] = batch['labels']
    
    # Reshape data 
    train_X = np.reshape(train_X, (50000, 3, 32, 32))
    #train_Y = create_one_hot(train_Y, 10)
    test_X = np.reshape(test_X, (10000, 3, 32, 32))
    #test_Y = create_one_hot(test_Y, 10)

    train_mean = np.mean(train_X, axis=(0,2,3), keepdims=True)
    train_std = np.std(train_X, axis=(0,2,3), keepdims=True)
    train_X = (train_X-train_mean)/train_std
    test_X = (test_X-train_mean)/train_std

    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y.astype(np.int64))
    test_X = torch.from_numpy(test_X)
    test_Y = torch.from_numpy(test_Y.astype(np.int64))
    
    #train_X = train_X.cuda()
    #train_Y = train_Y.cuda()
    #test_X = test_X.cuda()
    #test_Y = test_Y.cuda()

    #plt.imshow(np.transpose(train_X[0,:,:,:], (1,2,0)).astype(np.uint8)) 
    return (train_X, train_Y, test_X, test_Y)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = CNN()
    net.cuda()
    train_X, train_Y, test_X, test_Y = data_prepare()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    batch_size = 128
    num_train = 50000
    num_test = 10000
    num_ite_per_e = int(np.ceil(float(num_train)/float(batch_size)))
    full_ind = np.arange(num_train)
    rng = np.random.RandomState(1311) 
    
    for e in range(100):
        pdb.set_trace()
        running_loss = 0.0
        for i in range(num_ite_per_e):
            rng.shuffle(full_ind)
            optimizer.zero_grad()
            
            if (i+1)*batch_size <= num_train:
                batch_range = range(i*batch_size, (i+1)*batch_size)
            else:
                batch_range = range(i*batch_size, num_train)
            batch_range = full_ind[batch_range]
            batch_X = Variable(train_X[torch.from_numpy(batch_range)].cuda())
            batch_Y = Variable(train_Y[torch.from_numpy(batch_range)].cuda())
            pdb.set_trace()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            if i % 100 == 99:
                print('[%d, %d] loss: %.3f' % (e+1, e*num_ite_per_e+i+1, running_loss/100))
                running_loss = 0.0

