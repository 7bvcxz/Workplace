import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Activation, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

tf.device("/GPU:0")

# Use ResNet50 pre-trained model 
pre_model = tf.keras.applications.ResNet50(
                include_top=True, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000
            )

pre_model._layers.pop()
model = Model(pre_model.input, pre_model.layers[-1].output)

"""
# freezing
for layer in model.layers:
    if 'conv5' in layer.name:
        break
    layer.trainable = False
"""

x = model.output
x = Dense(2, activation='softmax', name='softmax')(x)
model = Model(model.input, x)

print(model.summary())

# data loader
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = train_datagen.flow_from_directory(
                    '../data',
                    target_size=(224, 224),
                    batch_size=30,
                    class_mode='categorical',
                    subset='training')

test_generator = train_datagen.flow_from_directory(
                    '../data',
                    target_size=(224, 224),
                    class_mode='categorical',
                    subset='validation')

# model compile
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])


history = model.fit_generator(train_generator,
                              steps_per_epoch=25,
                              epochs=150,
                              validation_data=test_generator)
                             
model_PATH = '../weight_convfc2.h5'
model.save(model_PATH)




