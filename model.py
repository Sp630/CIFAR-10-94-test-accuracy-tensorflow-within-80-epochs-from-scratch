import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.utils as utility
import tensorflow as tf
import datetime
import tensorflow.keras.backend as K

#Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalize the data so that it in the range 0 to 1, makes it easier for the NN to learn
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Set up parameters
lr = 0.1
lr_drop = 20
weight_decay = 1e-4
batch_size = 64

#Set up the learning rate scheduler function.
#During my own research I found this way of setting it  up to work very well
#Unfortunately I can't find the original post, I will link it if I find it
def lr_scheduler(epoch):
    print(epoch)
    return lr * (0.5 ** (epoch/lr_drop))

#Do some data augmentation to provide more diverse data to the NN
#It also helps to reduce overfitting
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

#Build the model
model = keras.Sequential([
    InputLayer(input_shape=(32, 32, 3)),
    #This Lambda layer is key, increasing the image size provides much better results: from 92% to 94%(2% increase)
    Lambda(lambda image: K.resize_images(image, 4, 4, 'channels_last')),
    Conv2D(filters= 64, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Conv2D(filters= 64, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(filters= 128, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters= 128, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(filters= 256, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters= 256, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters= 256, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(filters= 512, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters= 512, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters= 512, kernel_size=(3, 3), padding= 'same', kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Dropout(0.5),

    Flatten(),
    Dense(512, kernel_regularizer= keras.regularizers.l2(weight_decay)),
    BatchNormalization(),
    Activation('relu'),

    Dropout(0.5),
    Dense(10)

])

#compile the model
#my accuracy metric look wierd because I used SparseCategoricalCrossentropy
model.compile(optimizer= keras.optimizers.SGD(lr, 0.9, nesterov= True),
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits= True),
              metrics= keras.metrics.sparse_categorical_accuracy)


print(model.summary())

#use this if you want to plot a diagram in tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="your directory", histogram_freq=1)

#train the model, I only needed 77 epoch to acheive 94.09% accuracy,
#but this is the original code I used where I had set it to 600 epochs
model.fit(datagen.flow(x_train, y_train, batch_size= batch_size), steps_per_epoch= len(x_train) / batch_size, epochs= 600,
          validation_data=(x_test, y_test), verbose= 1, callbacks= [tensorboard_callback, keras.callbacks.LearningRateScheduler(lr_scheduler),
                                                                    keras.callbacks.ModelCheckpoint(
                                                                        monitor= 'val_sparse_categorical_accuracy',
                                                                        save_freq= 'epoch',
                                                                        save_best_only= True,
                                                                        save_weights_only= False,
                                                                        filepath= "trainedModel",
                                                                        mode= 'auto'
                                                                    )])
#evaluate the model
model.evaluate(x_test, y_test, verbose = 1, batch_size= 128)