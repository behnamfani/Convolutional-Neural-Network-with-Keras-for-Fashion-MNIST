# Convolutional-Neural-Network
A simple CNN using Tensorflow and Keras libraries and is trained with Fashion-MNIST dataset to predict images of T-shirts, shoes and etc.

The format of the images should be channels_first. Channels first mean that in a specific tensor (in this problem, input images), we have (Number_Of_Channels, Height, Width). 3 channels stand for RBG and 1 channel stands for grayscale. In this problem, the height and width of images are 28 * 28. 

This CNN consists of two convolutional layers, two max-pool layers, two dropout layers, and a fully connected network with two dense layers.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 28, 28)        320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 14, 14)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 14, 14)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 14, 14)        8224      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 7, 7)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 32, 7, 7)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1568)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               401664    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 412,778
Trainable params: 412,778
Non-trainable params: 0


