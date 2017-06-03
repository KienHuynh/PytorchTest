# PytorchTest

Requirements:
* Pytorch

## Test with cifar 10
To run cifar10.py:
* Download the data file .gz [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* Extract it in ../data/ so that the directory ../data/cifar-10-batches-py exists (or you can simply modify the path in cifar10.py)
* Run the code

Note: if you can't get CUDA supports to work, remove the lines:
```python
net.cuda()
```
And remove the .cuda() call in the following lines:
```
batch_X = Variable(train_X[torch.from_numpy(batch_range)].cuda())
batch_Y = Variable(train_Y[torch.from_numpy(batch_range)].cuda())
```

CUDA support check will be added later.

## Loading a pre-trained VGG
To run vgg_test.py:
* Download the numpy VGG weight [here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)
* Put it in ../data/pretrained/ or anywhere you want but remember to modify the path in vgg_test.py
* Change the path to the image you want to classify:
 ```python
 I = mpimg.imread('../../data/random_test_data/dog.jpg')
 ```
* Run the code
