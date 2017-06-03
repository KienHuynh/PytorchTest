import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import scipy.misc
import matplotlib.image as mpimg

import pdb

def VGG_preprocess(data):
    # Subtract the image batch by VGG mean
    VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = np.reshape(VGG_MEAN, (1, 3, 1, 1))
    return np.asarray(data-VGG_MEAN, dtype=np.float32)

class VGG(nn.Module):
    def __init__(self, data_path='../../data/pretrained/vgg16.npy'):
        super(VGG, self).__init__()
        # Load weight values from file
        data_dict = np.load(data_path).item()
        self.vgg = {} 
        self.vgg['conv1_1'] = nn.Conv2d(3, 64, 3, bias=True, padding=1)
        self.vgg['conv1_2'] = nn.Conv2d(64, 64, 3, bias=True, padding=1)
        
        self.vgg['conv2_1'] = nn.Conv2d(64, 128, 3, bias=True, padding=1)
        self.vgg['conv2_2'] = nn.Conv2d(128, 128, 3, bias=True, padding=1)
        
        self.vgg['conv3_1'] = nn.Conv2d(128, 256, 3, bias=True, padding=1)
        self.vgg['conv3_2'] = nn.Conv2d(256, 256, 3, bias=True, padding=1)
        self.vgg['conv3_3'] = nn.Conv2d(256, 256, 3, bias=True, padding=1)
        
        self.vgg['conv4_1'] = nn.Conv2d(256, 512, 3, bias=True, padding=1)
        self.vgg['conv4_2'] = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg['conv4_3'] = nn.Conv2d(512, 512, 3, bias=True, padding=1)

        self.vgg['conv5_1'] = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg['conv5_2'] = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg['conv5_3'] = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        
        self.vgg['fc6'] = nn.Linear(7*7*512, 4096)
        self.vgg['fc7'] = nn.Linear(4096, 4096)
        self.vgg['fc8'] = nn.Linear(4096, 1000)

        for k in self.vgg.keys():
            W = data_dict[k][0]
            b = data_dict[k][1]
            if (k == 'fc6'):
                W = np.reshape(W,(7,7,512,4096))
                W = np.transpose(W,(2,0,1,3))
                W = np.reshape(W,(7*7*512,4096))
                W = np.transpose(W, (1,0))

            elif (k[0:2] == 'fc'): 
                W = np.transpose(W, (1,0))

            if (k[0:4] == 'conv'):
                W = np.transpose(W, (3,2,0,1))
                # Flip the filter
                W_ = W[:,:,::-1,::-1]
                W = np.copy(W_)
        
            W_tensor, b_tensor = list(self.vgg[k].parameters())
            W_tensor.data = torch.from_numpy(W)
            b_tensor.data = torch.from_numpy(b)
            # pdb.set_trace()
        
        self.vgg['pool'] = nn.MaxPool2d(2,2)

    def forward(self, x): 
        x = F.relu(self.vgg['conv1_1'](x))
        x = F.relu(self.vgg['conv1_2'](x))
        x = self.vgg['pool'](x)
        
        x = F.relu(self.vgg['conv2_1'](x))
        x = F.relu(self.vgg['conv2_2'](x))
        x = self.vgg['pool'](x)

        x = F.relu(self.vgg['conv3_1'](x))
        x = F.relu(self.vgg['conv3_2'](x))
        x = F.relu(self.vgg['conv3_3'](x))
        x = self.vgg['pool'](x)
        
        x = F.relu(self.vgg['conv4_1'](x))
        x = F.relu(self.vgg['conv4_2'](x))
        x = F.relu(self.vgg['conv4_3'](x))
        x = self.vgg['pool'](x)

        x = F.relu(self.vgg['conv5_1'](x))
        x = F.relu(self.vgg['conv5_2'](x))
        x = F.relu(self.vgg['conv5_3'](x))
        x = self.vgg['pool'](x)
        x = x.view(-1, 7*7*512)
        x = self.vgg['fc6'](x)
        x = self.vgg['fc7'](x)
        x = self.vgg['fc8'](x)
        return x

if __name__ == "__main__":
    vgg = VGG()
    I = mpimg.imread('../../data/random_test_data/dog.jpg')
    I = np.asarray(scipy.misc.imresize(I, (224,224), 'bicubic'), dtype=np.float32)
    I = I[:,:,[2,1,0]]
    I = np.transpose(I, (2,0,1))
    I = np.reshape(I, (1, 3, 224, 224))
    I = VGG_preprocess(I)
    I = torch.from_numpy(I)
    I = Variable(I)

    f = open('synset.txt')
    lines = f.readlines()
    f.close()

    result = vgg.forward(I) 
    best = np.argmax(result.data.numpy()[0,:])
    print(lines[best])
