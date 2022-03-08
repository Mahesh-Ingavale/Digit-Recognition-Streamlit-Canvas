# Digit-Recognition-Streamlit-Canvas    
Digit Recognition or Identification [ Streamlit Canvas File ] Artificial Neural Network

#Part 1
# Installing Library
!pip install matplotlib-venn

#Part 2
# Inatalling Library
!apt-get -qq install -y libfluidsynth1

#Part 3
# Inatalling Library
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive

#Part 4
#Requirements
!apt-get -qq install -y graphviz && pip install pydot
import pydot

#Part 5
!pip install cartopy
import cartopy

#Part 6
!pip install streamlit --quiet
!pip install pyngrok==4.1.1 --quiet
!pip install streamlit-drawable-canvas --quiet
from pyngrok import ngrok

#Part 7
# Loading Training Model
%%writefile app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
model_new = keras.models.load_model('Training_Model_1_For_Digit_Recognition.hdf5')

# Create a canvas component
import numpy as np
a = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  97,  84,
         97, 110, 108,  87,  97,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  19, 255, 199,
        183, 142, 161, 200, 213,  32,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  72, 174, 173,
        158, 131, 132, 176, 176,  70,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 142, 180, 163,
        120, 180, 126, 139, 168, 122,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 156, 180, 162,
        171, 170, 183, 163, 169, 144,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 159, 177, 163,
        174, 166, 173, 170, 163, 148,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 160, 182, 160,
        168, 175, 167, 157, 168, 139,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 143, 185, 161,
        171, 184, 176, 160, 175, 131,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 143, 188, 155,
        172, 182, 180, 161, 182, 127,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 127, 170, 168,
        175, 211, 191, 163, 184, 113,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  78, 132, 175,
        177, 129, 203, 161, 187, 102,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  91, 166, 163,
        190,  63, 203, 159, 189,  94,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  98, 126, 166,
        192,  65, 198, 167, 193,  84,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  90, 127, 170,
        198,  61, 200, 174, 198,  71,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  93, 141, 151,
        207,  55, 196, 171, 196,  58,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  88, 137, 136,
        210,  54, 198, 173, 199,  46,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  72, 125, 131,
        214,  55, 199, 176, 200,  38,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  56, 146, 126,
        210,  56, 200, 175, 200,  37,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  40, 237, 171,
        195,  64, 199, 174, 203,  36,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  23, 190, 168,
        192,  70, 196, 171, 203,  30,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  15, 188, 171,
        196,  70, 198, 173, 202,  21,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 187, 172,
        197,  71, 199, 174, 203,  15,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1, 183, 175,
        198,  73, 195, 174, 205,  13,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 186, 177,
        200,  78, 191, 175, 171,   8,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 181, 177,
        198,  78, 189, 176, 166,   4,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 178, 182,
        182, 107, 184, 175, 193,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 173, 200,
        206,  84, 196, 196, 133,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68, 207,
        182,   0, 175, 195,  34,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype='uint8')

op = model_new.predict(a.reshape(1,28,28))
label = "Digit Recognition Using Artificial Neural Network [ANN]"
label = label.split()
label

z = label[np.argmax(op)]

st.title(z)
#Overwriting The File "app.py"

#Part 8
!pip install streamlit --quiet
!pip install ipykernel 5.5.5 --quite
!pip install pyngrok==4.1.1 --quiet
from pyngrok import ngrok

#Part 9
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

numpy_data = np.random.randn(100,3,224,224) # 10 samples, image size = 224 x 224 x 3
numpy_target = np.random.randint(0,5,size=(100))

dataset = MyDataset(numpy_data, numpy_target)


#Part 10
#Load Training Model 1
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import webbrowser 

def load_model():
    model_name = "Training_Model_1_For_Digit_Recognition.hdf5"

    # Model reconstruction from JSON file
   with open( model_name + '.json', 'r') as f:
        model = model_from_json(f.read())
        
        
#Part 11
#Import Libraries
import numpy as np
import keras
import matplotlib.pyplot as plt


#Part 12
import tensorflow as tf
print(tf.__version__)

#Part 13
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Part 14
def show(idx):
    print(y_train[idx])
    plt.imshow(x_train[idx])
  
#Part 15
show(2)

#Part 16
#Model Execution
np.random.seed(23)
tf.random.set_seed(23)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0003),
              metrics=['accuracy'])
model.summary()


#Part 16
#Run Model And Chech Accuracy
model.fit(x_train, y_train, batch_size=32, epochs=20)

#Part 17
#Digit Recognition Based On ANN
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

#Using TenserFlow and Keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Part 18
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#Part 19
X_train.shape

#Part 20
plt.imshow(X_train[1,:,:],cmap = 'gray')

#Part 21
plt.imshow(X_test[0,:,:],cmap='gray')










