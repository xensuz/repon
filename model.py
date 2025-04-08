import os  # for accessing files
import pandas as pd
import numpy as np
import cv2  # for image prosessing
from PIL import Image #PIL -> pillow libary used for image processing libraries
from sklearn.model_selection import train_test_split

with_mask_files=os.listdir('C:/Users/msi/OneDrive/文档/data science/DL Project/data/with_mask') # create list contain files with mask
print(with_mask_files[:5])  # print the first five elements
print(with_mask_files[-5:]) # print the last five elements of the list

without_mask_files=os.listdir('C:/Users/msi/OneDrive/文档/data science/DL Project/data/without_mask') # create list contain files without mask
print(without_mask_files[:5])  # print the first five elements
print(without_mask_files[-5:])  # print the last five elements of the list

#creating the labels
with_mask_labels = [0]*len(with_mask_files)
without_mask_labels = [1]*len(without_mask_files)

print(with_mask_labels[:5])
print(without_mask_labels[:5])

labels = with_mask_labels + without_mask_labels # adding the to list

print(len(labels))
print(labels[0:5])
print(labels[-5:])

def convert_images_to_numpy(image_dir, target_size=(128, 128)):
    data = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            with Image.open(img_path) as image:
                image = image.resize(target_size)  # resize image
                image = image.convert('RGB')       # convert to RGB VV
                image_array = np.array(image)      # convert to numpy array
                data.append(image_array)
    return data

# Paths to the directories containing images
with_mask_path = 'C:/Users/msi/OneDrive/文档/data science/DL Project/data/with_mask/'
without_mask_path = 'C:/Users/msi/OneDrive/文档/data science/DL Project/data/without_mask/'

# Convert images in both directories to numpy arrays
data_with_mask = convert_images_to_numpy(with_mask_path)
data_without_mask = convert_images_to_numpy(without_mask_path)

# Combine the data from both directories
data = data_with_mask + data_without_mask

#converting image list and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)

type(X)
type(Y)

Y

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
print(X.shape, x_train.shape, x_test.shape)

#scaling the data

x_train_scaled =  x_train/255.0   #to change the value from 0 to 1


x_test_scaled = x_test/255.0

import tensorflow as tf
from tensorflow import keras

num_of_classes = 2

# Convert integer class labels to one-hot encoded vectors, Y_train  should contains integer class labels

Y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_of_classes)


Y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_of_classes)

from tensorflow import keras

num_of_classes = 2
input_shape = (128, 128, 3)

model = keras.Sequential()

# First Conv Block
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second Conv Block
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Third Conv Block
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Global Average Pooling
model.add(keras.layers.GlobalAveragePooling2D())

# Fully Connected Layers
model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dropout(0.5))

# Output Layer
#model.add(keras.layers.Dense(1, activation='sigmoid'))  # For binary classification
# Use softmax if you have one-hot encoded labels:
model.add(keras.layers.Dense(num_of_classes, activation='softmax'))



# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change to 'categorical_crossentropy' for one-hot labels
              metrics=['accuracy'])
# Your existing code up to model.fit() remains the same until training
# After training, add this to save the model
model.save('mask_detector.h5')

# Complete training code with save
history = model.fit(x_train_scaled, Y_train_one_hot, 
                   validation_split=0.1, 
                   epochs=15)
model.save('mask_detector.h5')

