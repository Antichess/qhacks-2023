# Import libraries
import tensorflow as tf
from tensorflow import keras
# For image manipulation
import PIL
import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# For linear algebra
import numpy as np
import matplotlib.pyplot as plt



img_height, img_width = 480, 640
""" class_names = ['Safe_driver', 'Texting_right', 'Calling_right', 'Texting_left', 'Calling_left',
               'Infotainment', 'Drinking', 'Reaching', 'Scratching', 'Head_turned'] """

class_names = ['Calling_left', 'Calling_right', 'Drinking', 'Head_turned', 'Infotainment',
               'Reaching', 'Safe_driver', 'Scratching', 'Texting_left', 'Texting_right']


model = tf.keras.models.load_model('model.h5', compile=False)

# Load the image
img = keras.preprocessing.image.load_img(
    #"imgs_reduced/testing/test/img_4895.jpg", target_size=(img_height, img_width)
    
    "C:/git repos/qhacks-2023/pyqt/test_image.png", target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Predict the image
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Print the results
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)