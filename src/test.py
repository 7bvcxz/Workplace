import tensorflow as tf

# Load MNIST dataset and scale values to [0 1].
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

model = tf.keras.models.load_model("../weight.h5")

#Test the model
results = model.predict(x_test)
