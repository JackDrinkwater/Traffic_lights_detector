import numpy as np
import pandas as pd 
from tensorflow.keras.losses import binary_crossentropy, GUoILoss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.metrics import TruePositives, FalsePositives, FalseNegatives

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3


def custom_loss(y_true, y_pred):
    class_loss = binary_crossentropy(y_true[:,:-1], y_pred[:, :-1])
    # need make Euclidian distance loss here
    reg_loss = GUoILoss(y_true[:, -1], y_pred[:, -1])
    
    return class_loss * y_true[:, -1] + 2 * reg_loss

def make_model():
    input_layer = Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    layer_1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    layer_2 = Conv2D(32, (3, 3), activation='relu')(layer_1)
    layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_2)

    
    layer_3 = Conv2D(64, (3, 3), activation='relu')(layer_2)
    layer_3 = MaxPooling2D(pool_size=(2, 2))(layer_3)
    
    layer_4 = Flatten()(layer_3)
    layer_4 = Dense(1000, activation='relu')(layer_4)
    
    layer_5 = Dense(200, activation='relu')(layer_4)
    
    out_1 = Dense(1, activation='sigmoid')(layer_4)
    out_2 = Dense(4, activation='linear')(layer_4)
    
    model = Model(inputs=input_layer, outputs=[out_1,out_2])
    model.compile(optimizer = "rmsprop", loss = custom_loss)
    
    return model