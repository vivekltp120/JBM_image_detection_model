import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img






datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


img_path='/home/viv/Per/MachineLearning_2019/tensorflow_project/YE358311_Fender_apron/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/IMG20180905144824.jpg'
img=load_img(img_path)
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
plt.imshow(x)
# plt.show()
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='jbm', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely