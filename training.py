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
 
""" This model predicts total electrical output from wind farms """
 
def scale_speed(input):
    #scale speed to range of [0,1]
    maximum = max(input)
 
    input = input/maximum
    input = [x / maximum for x in input]
def scale_direction(input):
    #scale to range of [0,2]
    #new_input = [i * 5 for i in input]
    input = np.sin(input) + 1
    return input
 
df=pd.read_csv('Demo_Data.csv')
speed = df['Wind Speed (m/s)']
speed = speed/max(speed)
 
direction = df['Wind Direction (°)']
direction = scale_direction(direction)
 
y_label = df['LV ActivePower (kW)']
df = pd.DataFrame(speed)
df2 = pd.DataFrame(y_label)
 
#scaling
scale_speed(inputSpeed)
scale_direction(inputDir)
inputs = tf.keras.Input(shape=(1,))
 
 
 
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation = 'relu')(x)
end = layers.Dense(3605, activation="softmax")(x)
 
 
model = tf.keras.Model(inputs=inputs, outputs=end)
 
 
# Compile the model and predict
 
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
 
model.fit(
    df, df2,
    epochs=10, batch_size = 32)
 
line = np.polyfit(speed, df2, 1, full=False)
print(line)
 
 
#from sklearn.metrics import r2_score
#preds = model.predict(df)
#preds = preds.transpose()
 
#rs = r2_score(df2,preds)
