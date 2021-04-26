import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = tf.keras.models.load_model("../weight_convfc.h5")

#Test the model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
                    '../data/Train',
                    target_size=(224, 224),
                    batch_size=30,
                    class_mode='categorical')

results = model.predict(test_generator)
loss, accuracy = model.evaluate(test_generator)

print(np.argmax(results, axis=1))
print(accuracy)
