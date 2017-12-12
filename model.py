### CHRASMUS (Claus H. Rasmussen)
### model.py
###

### IMPORT SECTION
import time
import random
import os
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.backend import tf as ktf

### CONSTANTS
EPOCHS = 3
ch, row, col = 3, 64, 64  # Trimmed image format

### READ THREE DATA FILES

samples = []

### contains to rounds of center driving
with open('/Users/claushrasmussen/Desktop/Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
print('Number of observations: ' + str(len(samples)))
print(time.ctime())

### Contains a reverse drive through the curves to the bridges
with open('./Data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
print('Number of observations: ' + str(len(samples)))
print(time.ctime())

### Contains a reverse drive through the curves to the bridges and back to start
### Contains a lot of recovery driving
with open('./Data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print('Number of observations: ' + str(len(samples)))
print(time.ctime())

### Split the images into a training images and validation images, using a 80/20 pct split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

### The generator function, which will deliver images on-the-fly instead of reading them all into memory.
### Incredibly slow when using a batch size of 32. 11 hour for three epochs!
### TODO : experience with larger batch_sizes
def my_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # commented code, not for use on a local Mac Pro
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                #center_image = cv2.imread(name)
                center_image = cv2.imread(batch_sample[0])
                # abandommed code, gave to many errors
                # CHRASMUS : resizing to 64x64
                #image_array = cv2.resize(image_array, (64, 64))
                # CHRASMUS : Make sure to use the RGB colorspace in the model.
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                # CHRASMUS : Use only 50 pct of images with low steering angle, less than 0.75
                # this piece of code remove the biased 'driving straight' images
                if center_angle >= 0.75:
                    chance = random.randint(1,100)
                    if chance <= 50:
                        images.append(center_image)
                        angles.append(center_angle)
                        # CHRASMUS : augmentation = add flipped image
                        images.append(cv2.flip(center_image,1))
                        angles.append(center_angle*-1.0)
                else:
                    images.append(center_image)
                    angles.append(center_angle)
                    # CHRASMUS : augmentation = add flipped image
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

### CHRASMUS : commented malfunction code
#def resize_img(input):
#    return ktf.image.resize_images(input, (row, col))
# DOESN'T WORK :-( re-size inside the model
# model.add(Lambda(lambda x: resize_img(x)))

# compile and train the model using the generator function
train_generator = my_generator(train_samples, batch_size=32)
validation_generator = my_generator(validation_samples, batch_size=32)

# This is the Nvidia model, with two extra Dropout layers to handle overfitting.
model = Sequential()

### CHRASMUS : Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
### CHRASMUS : trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))
### CHRASMUS : reduce the image size by half
### commented, gave errors in the model
### TODO : uncomment and fix the model accordingly
#model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=None))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

### use an Adam optimizer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), verbose = 1, epochs=EPOCHS)

### Save the model
print(time.ctime())
model.save('model.h5')
print('model saved')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
