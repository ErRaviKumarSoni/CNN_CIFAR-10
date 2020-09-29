#CNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense,Conv2D, MaxPooling2D, Dropout

#Load the dataset
from tensorflow.keras.datasets import cifar10
#Convert data into Test Train
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Get Classes

class_names =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Find Describe

X_train = X_train/255
X_train
X_test = X_test/255
X_test

X_train.shape
X_test.shape
y_train

#Build the CNN model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [32,32,3]))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [32,32,3]))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='Valid'))
model.add(Dropout(0.5))
model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax')) # Softmax used because binary values not found 

model.summary()

#Compile the model
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy'])

history = model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(X_test, y_test)) 

#Plot training & validation 'sparse_categorical_crossentropy' values
import matplotlib
history.history
epoch_range = range(1, 11)
plt.plot(epoch_range, history.history['sparse_categorical_crossentropy'])
plt.plot(epoch_range, history.history['val_sparse_categorical_crossentropy'])
plt.title('Model accuracy')
plt.xlabel("Accuracy")
plt.ylabel(ylabel='Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

# Plot traning & validation loss values
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title('Model loss')
plt.xlabel("Epoch")
plt.ylabel(ylabel='Loss')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

#Confusion matrix

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)
y_pred

y_test

mat = confusion_matrix(y_test, y_pred)
mat

plot_confusion_matrix(conf_mat=mat, figsize=(8, 8), class_names=class_names, show_normed=True)
