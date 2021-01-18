import numpy as np
import pandas as pd 
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.metrics import TruePositives, FalsePositives, FalseNegatives
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3


def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [1, 5])
    y_pred = tf.reshape(y_pred, [1, 5])

    class_loss = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
    # need make Euclidian distance loss here
    reg_loss = mse(y_true[:, 1:5], y_pred[:, 1:5])
    
    return class_loss * y_true[:, -1] + 2 * reg_loss

def make_model():
    input_layer = Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    
    x = Dense(200, activation='relu')(x)
    
    out_1 = Dense(1, activation='sigmoid')(x)
    out_2 = Dense(4, activation='linear')(x)

    output_layer = Concatenate()([out_1, out_2])
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer = Adam(learning_rate=0.0001), loss = custom_loss)
    
    return model