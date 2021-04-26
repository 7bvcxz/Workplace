import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Activation, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

tf.device('/CPU:0')

"""
# Use ResNet101 pre-trained model 
model = tf.keras.applications.ResNet101(
                include_top=True, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000
            )


#model.layers.pop()
print(model.summary())


model.save(pre_model_PATH)

"""

pre_model_PATH = '../pretrained.h5'
pre_model = tf.keras.models.load_model(pre_model_PATH)
pre_model._layers.pop()

model = Model(pre_model.input, pre_model.layers[-1].output)

# freezing
for layer in model.layers:
    if 'conv5' in layer.name and 'block3' in layer.name:
        break
    layer.trainable = False

x = model.output
x = Dense(2, activation='softmax', name='softmax')(x)
model = Model(model.input, x)

print(model.summary())

# data loader
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                    '../data/Train',
                    target_size=(224, 224),
                    batch_size=30,
                    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(
                    '../data/Test',
                    target_size=(224, 224),
                    batch_size=30,
                    class_mode='categorical')


# model compile
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])


history = model.fit_generator(train_generator,
                              steps_per_epoch=25,
                              epochs=30,
                              validation_data=test_generator,
                              validation_steps=5)
                             
model_PATH = '../weight.h5'
model.save(model_PATH)




