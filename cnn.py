from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
import os
import tensorflow as tf
# from tensorflow.keras import initializers

# initializer = tf.keras.initializers.HeNormal()
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3),  activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer

# classifier.add(Conv2D(128, (3, 3),  activation = 'relu'))
# classifier.add(BatchNormalization())
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.1))

classifier.add(Dense(units = 64, activation = 'relu'))
# classifier.add(Dropout(0.1))
classifier.add(Dense(units=1,activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('DogsNCats/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('DogsNCats/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# model = classifier.fit_generator(training_set,
#                          #steps_per_epoch = 8000,
#                          epochs = 40,
#                          validation_data = test_set)   
#                          #validation_steps = 2000)

# newpath = 'model'
# if not os.path.exists(newpath):
#     os.makedirs(newpath)

classifier.save("dog_cat_model.h5")
print("Saved model to disk")

# # Part 3 - Making new predictions

from tensorflow import keras

model = keras.models.load_model('dog_cat_model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cat_test2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) #DogsNCats\validation\dogs\dog.2001.jpg,DogsNCats/train/cats/cat.1.jpg,DogsNCats/cat.2002.jpg,DogsNCats/train/dogs/dog.2.jpg
test_image = np.expand_dims(test_image, axis = 0) #DogsNCats\train\dogs\dog.1.jpg
result = model.predict(test_image)
print(result)
print(training_set.class_indices)
if result[0][0] == 1 :
    prediction = 'dog'
    print(prediction)
    print(result[0][0])
else:
    prediction = 'cat'
    print(prediction)
 