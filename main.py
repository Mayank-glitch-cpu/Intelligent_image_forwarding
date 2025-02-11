import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from zipfile import ZipFile
from PIL import Image as im
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

# Set local path to images (ensure the images exist in this directory)
path = './animals/'  # Updated to local directory
categories = ['dogs', 'panda', 'cats']

# Display some pictures
for category in categories:
    fig, _ = plt.subplots(3, 4)
    fig.suptitle(category)
    for k, v in enumerate(os.listdir(path + category)[:12]):
        img = plt.imread(path + category + '/' + v)
        plt.subplot(3, 4, k + 1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()

# Preprocess data and label inputs
data = []
labels = []
imagePaths = []
HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path + category):
        imagePaths.append([path + category + '/' + f, k])

import random
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data.append(image)
    label = imagePath[1]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split dataset into train and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = np_utils.to_categorical(trainY, 3)

# Define model architecture
model = Sequential()
model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1)

# Evaluate model on test data
pred = model.predict(testX)
predictions = argmax(pred, axis=1)
cm = confusion_matrix(testY, predictions)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + categories)
ax.set_yticklabels([''] + categories)

for i in range(3):
    for j in range(3):
        ax.text(i, j, cm[j, i], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

# Save model
model.save('image_forwarding_model.h5')
