from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import glob
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import sys

d_path='./Data/validation/YE358311_Crack_and_Wrinkle_defect/'
h_path='./Data/validation/YE358311_Healthy/'
td_path='./Data/train/YE358311_Crack_and_Wrinkle_defect/'
th_path='./Data/train/YE358311_Healthy/'
# Load models
model = load_model('./detect_defect_v9.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

path_for_source=sys.argv[1]

#Load images
# dimensions of our images.
img_width, img_height = 250, 250
fig = plt.figure(figsize=(14, 14))
for cnt,testimagepath in enumerate(glob.glob(pathname=path_for_source+'*',),1):
    image=load_img(path=testimagepath,target_size=(img_width, img_height))
    img = cv2.imread(testimagepath)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height,3])

    class_of_image = model.predict_classes(img)

    print(os.path.split(testimagepath)[1],end=' ')
    if class_of_image[0][0]==0:
     print('- image is defected')
    else:
     print('- image is healthy')





