# Updated imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

# Pre-initialized terms
path = '/Users/Bingumalla Likith/Desktop/Projects/Sudoku_Solver/myData'
batchSizeVal = 200
epochsVal = 30

####### Model #######

# Retrieve the images and store them in the form of numpy array
myList = os.listdir(path)
noOfClasses = len(myList)

images = []
classNo = []

for x in range(0, noOfClasses):
    EachPicList = os.listdir(path + '/' + str(x))
    for y in EachPicList:
        currImage = cv2.imread(path + '/' + str(x) + '/' + y)
        currImage = cv2.resize(currImage, (32, 32))
        images.append(currImage)
        classNo.append(x)

images = np.array(images)
classNo = np.array(classNo)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

numberOfSamples = []
for x in range(0, noOfClasses):
    numberOfSamples.append(len(np.where(y_train == x)[0]))

def PreProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

x_train = np.array(list(map(PreProcessing, x_train)))
x_test = np.array(list(map(PreProcessing, x_test)))
x_validation = np.array(list(map(PreProcessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.3, shear_range=0.1, rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# LeNet Architecture based Model
def MyModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    NoOfNodes = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(NoOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = MyModel()
history = model.fit(dataGen.flow(x_train, y_train, batch_size=batchSizeVal),
                    steps_per_epoch=len(x_train)//batchSizeVal,
                    epochs=epochsVal,
                    validation_data=(x_validation, y_validation),
                    shuffle=1)
pd.DataFrame(history.history).plot(figsize=(20, 15))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test score = ", score[0])
print("Test accuracy = ", score[1])

# Save the model
model.save("New_Model_trained.h5")
