# Image recognition app using tensorflow and keras to determine if a webcam video is one of 10 categories

# Import libraries
import tensorflow as tf
from tensorflow import keras
# For image manipulation
import PIL
import pathlib
# For linear algebra
import numpy as np
import matplotlib.pyplot as plt

# Class names for the 10 different categories identifying whether a driver is impaired or not
class_names = ['Safe_driver', 'Texting_right', 'Calling_right', 'Texting_left', 'Calling_left',
               'Infotainment', 'Drinking', 'Reaching', 'Scratching', 'Head_turned']

num_classes = 10

# Image paths
data_dir = pathlib.Path('imgs')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
print(data_dir)
test_dir = pathlib.Path('imgs/testing')
train_dir = pathlib.Path('imgs/train')

batch_size = 32
# Image size used throughout the process
img_height, img_width = 480, 640

# Number of epochs to train the model
epochs = 15

# Load the data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    labels="inferred",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    labels="inferred",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# For testing purposes
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Create the model
model = tf.keras.Sequential([
    # Normalization layer
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    # Convolutional layers
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 128 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit_generator(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print('Test accuracy:', test_acc)

# Save the model
model.save('model.h5')

# Plot the accuracy and loss over time
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the image
img = keras.preprocessing.image.load_img(
    "imgs/testing/Scratching/Scratching_00001.jpg", target_size=(img_height, img_width)
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
