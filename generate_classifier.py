#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:36:01 2020

@author: francesco
"""

from dl.DynamicDLModel import DynamicDLModel
import numpy as np # this is assumed to be available in every context

def class_unet():
    from keras.layers import Layer, InputSpec
    from keras import initializers, regularizers, constraints
    from keras.activations import softmax
    from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Concatenate, Lambda, ZeroPadding2D, Activation, Reshape, Add
    from keras.models import Sequential, Model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, Callback
    from keras.utils import plot_model, Sequence

    
    inputs=Input(shape=(128,128))
    reshape=Reshape((128,128,1))(inputs)

    reg=0.01
    
    #reshape=Dropout(0.0)(reshape)   ## Hyperparameter optimization only on visible layer
    Level1_l=Conv2D(filters=32,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(reshape)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    Level1_l_shortcut=Level1_l#Level1_l#
    Level1_l=Activation('relu')(Level1_l)
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)#(Level1_l)# ##  kernel_initializer='glorot_uniform' is the default
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Activation('relu')(Level1_l)
    #Level1_l=Dropout(0.5)(Level1_l)   
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Add()([Level1_l,Level1_l_shortcut])
    Level1_l=Activation('relu')(Level1_l)


    Level2_l=Conv2D(filters=64,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    Level2_l_shortcut=Level2_l
    Level2_l=Activation('relu')(Level2_l)
    #Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=ZeroPadding2D(padding=(1,1))(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Activation('relu')(Level2_l)
    #Level2_l=Dropout(0.5)(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Add()([Level2_l,Level2_l_shortcut])
    Level2_l=Activation('relu')(Level2_l)
    
    
    Level3_l=Conv2D(filters=128,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    Level3_l_shortcut=Level3_l
    Level3_l=Activation('relu')(Level3_l)
    #Level3_l=ZeroPadding2D(padding=(1,1))(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Activation('relu')(Level3_l)
    #Level3_l=Dropout(0.5)(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Add()([Level3_l,Level3_l_shortcut])
    Level3_l=Activation('relu')(Level3_l)
    
    
    Level4_l=Conv2D(filters=256,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    Level4_l_shortcut=Level4_l
    Level4_l=Activation('relu')(Level4_l)
    #Level4_l=ZeroPadding2D(padding=(1,1))(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Activation('relu')(Level4_l)
    #Level4_l=Dropout(0.5)(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Add()([Level4_l,Level4_l_shortcut])
    Level4_l=Activation('relu')(Level4_l)


    Level5_l=Conv2D(filters=512,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    Level5_l_shortcut=Level5_l
    Level5_l=Activation('relu')(Level5_l)
    #Level5_l=BatchNormalization(axis=-1)(Level5_l) 
    #Level5_l=ZeroPadding2D(padding=(1,1))(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Activation('relu')(Level5_l)
    #Level5_l=Dropout(0.5)(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Add()([Level5_l,Level5_l_shortcut])
    Level5_l=Activation('relu')(Level5_l)

    Level_f=Flatten()(Level5_l)
    output=Dense(2,activation='softmax',kernel_regularizer=regularizers.l2(reg))(Level_f)
    #output=Dense(2,kernel_regularizer=regularizers.l2(reg))(Level_f)
    #output=Lambda(lambda x : softmax(x,axis=-1))(output)
    ##output=Dense(1,activation=contracted_sigmoid,kernel_regularizer=regularizers.l2(reg))(Level_f) #loss='binary_crossentropy'
    #output=BatchNormalization(axis=-1)(output)
    model=Model(inputs=inputs,outputs=output)
    return model

def class_apply(modelObj: DynamicDLModel, data: dict):
    from dl.common.padorcut import padorcut
    from scipy.ndimage import zoom
    try:
        np
    except:
        import numpy as np
    
    LABELS_DICT = {
        0: 'Thigh',
        1: 'Leg'
        }
    
    MODEL_RESOLUTION = np.array([1.037037, 1.037037])*432/128 # the resolution is for an image of 432x432 but we process 128x128 images
    MODEL_SIZE = (128,128)
    netc = modelObj.model
    resolution = np.array(data['resolution'])
    zoomFactor = resolution/MODEL_RESOLUTION
    img = data['image']
    img = zoom(img, zoomFactor) # resample the image to the model resolution
    img = padorcut(img, MODEL_SIZE)
    categories = netc.predict(np.expand_dims(img,axis=0))
    value = categories[0].argmax()
    try:
        return LABELS_DICT[value]
    except KeyError:
        return None
    

model = class_unet()
model.load_weights('weights/weights_cosciagamba.hdf5')
weights = model.get_weights()

modelObject = DynamicDLModel('3f2a8066-007d-4c49-96b0-5fb7a703f6d0',
                             class_unet,
                             class_apply,
                             weights = weights,
                             timestamp_id="1603281030"
                             )

with open('models/Classifier_1603281030.model', 'wb') as f:
    modelObject.dump(f)
