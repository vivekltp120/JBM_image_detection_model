'''
"Building powerful image classification models using very little data"
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created Healthy/ and YE358311_Crack_and_Wrinkle_defect/ subfolders inside train/ and validation/
- put the YE358311_Healthy pictures 139 in data/train/YE358311_Healthy
- put the YE358311_Healthy pictures 10  in data/validation/YE358311_Healthy
- put the YE358311_Crack_and_Wrinkle_defect pictures 111 in data/train/YE358311_Crack_and_Wrinkle_defect
- put the dog pictures 10  in data/validation/YE358311_Crack_and_Wrinkle_defect
So that we have 111 training examples for each class, and 10 validation examples for each class.
In summary, this is our directory structure:
```

```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import pandas as pd

# dimensions of our images.
img_width, img_height = 250, 250

train_data_dir = 'Data/train'
validation_data_dir = 'Data/validation'
nb_train_samples = 101
nb_validation_samples = 10
epochs = 100
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary')
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary')



model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save('detect_defect_v9.h5')
