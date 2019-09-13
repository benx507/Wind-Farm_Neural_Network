import tensorflow as tf
import keras
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import argparse
import locale
import os
import MLP
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
 
""" This model predicts total electrical output from wind farms """
 
#SCALING FUNCTIONS

def scale_speed(input):
    #scale speed to range of [0,1]
    maximum = max(input)
 
    input = input/maximum
    input = [x / maximum for x in input]

def scale_direction(input):
    #scale to range of [0,2]
    new_input = [i * 5 for i in input]
    input = np.sin(input) + 1
    return input
 
df=pd.read_csv('Demo_Data.csv')


#CLEANING THE CSV DATA

# Making a list of missing value types
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("Demo_Data.csv", na_values = missing_values)

#Loop through the Wind Speed column. try and turn the entry into an integer, 
#if integer, enter missing value. If not integer, then string, so keep going.
cnt=0
for row in df['Wind Speed (m/s)']:
    try:
        int(row)
        df.loc[cnt, 'Wind Speed (m/s)']=np.nan
    except ValueError:
        pass
    cnt+=1

#If column missing value of types above, replace with 0. 
df['Wind Speed (m/s)'].fillna(125, inplace=True)
df['Wind Direction (°)'].fillna(125, inplace=True)

#SCALE DATA INPUTS

speed = df['Wind Speed (m/s)']
speed = speed/max(speed)
 
direction = df['Wind Direction (°)']
direction = scale_direction(direction)
 
y_label = df['LV ActivePower (kW)']
df = pd.DataFrame(speed)
df2 = pd.DataFrame(y_label)
 
scale_speed(inputSpeed)
scale_direction(inputDir)
inputs = tf.keras.Input(shape=(1,))
 
 
#CREATE MLP
 
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation = 'relu')(x)
end = layers.Dense(3605, activation="softmax")(x)


#OPTIONAL IMPLEMENTATION: Sequential Model w/ Dropout layers
y = Sequential()
y.add(Dense(50, input_dim = X_train.shape[1], 
                kernel_initializer = 'random_uniform', 
                activation = 'sigmoid'))
y.add(Dropout(0.2))
y.add(Dense(100, activation = 'relu'))
y.add(Dropout(0.5))
y.add(Dense(100, activation = 'relu'))
y.add(Dropout(0.5))
y.add(Dense(25, activation = 'relu'))
y.add(Dropout(0.2))
y.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))

 
 
model = tf.keras.Model(inputs=inputs, outputs=end)
 
 
# Compile the model and predict
 
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
 
model.fit(
    df, df2,
    epochs=10, batch_size = 32)
 

preds = model.predict(df)
preds = preds.transpose()
 
rs = r2_score(df2,preds)

print("Neural Network Accuracy: " + str(round(r2, 3)))
 
