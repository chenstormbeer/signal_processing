import tensorflow as tf
from tensorflow import keras
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
tf.enable_eager_execution()

def load_mnist(kind='train'):
    images_path='MNIST_data/{}-images-idx3-ubyte'.format(kind)
    labels_path='MNIST_data/{}-labels-idx1-ubyte'.format(kind)
    with open(labels_path,'rb')as labpath:
        magic,num=struct.unpack('>II',labpath.read(8))
        labels=np.fromfile(labpath,dtype=np.uint8)

    with open(images_path,'rb')as imagpath:
        magic,num,rows,columns=struct.unpack('>IIII',imagpath.read(16))
        images=np.fromfile(imagpath,dtype=np.uint8).reshape(len(labels),28,28)

    return images,labels

train_images,train_labels=load_mnist()
test_images,test_labels=load_mnist('t10k')

train_images=train_images/255.0
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(32,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']
              )
model.fit(train_images,train_labels,batch_size=100,epochs=20)
loss,acc=model.evaluate(test_images,test_labels)
print(acc)
