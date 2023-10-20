import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.models import Sequential
from keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, InputLayer, Reshape, Conv1D, MaxPool1D, SeparableConv2D
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    'cell_images',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    'cell_images',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
)

# Create a convolutional neural network model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc * 100:.2f}%')


model.save('malaria_model.h5')


import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('malaria_model.h5')  # Replace with the path to your trained model file

# Load and preprocess the new image you want to predict
new_image_path = "cell_images/Parasitized/C37BP2_thinF_IMG_20150620_131423a_cell_94.png"  # Replace with the path to your new image
img = image.load_img(new_image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image

# Make the prediction
prediction = model.predict(img_array)

# Interpret the prediction
if prediction[0][0] > 0.5:
    print("The image contains a parasitized cell (Malaria positive).")
else:
    print("The image contains an uninfected cell (Malaria negative).")

    
# Display the image
plt.imshow(img)
plt.axis('off')  # Hide the axis labels
plt.show()



