#Densely Connected Convolutional Network (DenseNet)
This repository contains the caffe version code for the paper [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993). 

For a brief introduction of DenseNet, see our original [Torch implementation](https://github.com/liuzhuang13/DenseNet).

##Note
This code is not the code we use to obtain the results in the original paper, the details (such as input preprocessing, data augmentation, training epochs) may be different. To reproduce the results reported in our paper, see our original [Torch implementation](https://github.com/liuzhuang13/DenseNet#introduction) .

##Results
The default setting (L=40, k=12, dropout=0.2) in the code yields a 7.09% error rate on CIFAR10 dataset (without any data augmentation).


##Usage 
0. Get the CIFAR data prepared following the [Caffe's official CIFAR tutorial] (http://caffe.berkeleyvision.org/gathered/examples/cifar10.html).
1. make\_densenet.py contains the code to generate the network and solver prototxt file. First change the data path in function make\_net() and preprocessing mean file in function densenet() to your own path of corresponding data file.
2. By default make\_densenet.py generates a DenseNet with Depth L=40, Growth rate k=12 and Dropout=0.2. To experiment with different settings, change the code accordingly (see the comments in the code). Example prototxt files are already included. Use ```python densenet_make.py``` to generate new prototxt files.
3. Change the caffe path in train.sh. Then use ```sh train.sh``` to train a DenseNet.

##Contact
liuzhuangthu at gmail.com  
gh349 at cornell.edu   
Any discussions, suggestions and questions are welcome!



