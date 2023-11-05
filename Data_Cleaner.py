'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-04 10:47:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-04 10:53:32
FilePath: \newrgzn\Data_Cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import sys
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import  image
def load_image(file_path):
    try:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0  # 标准化
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
def clean_data(train_bread_dir,train_dogs_dir,validation_bread_dir,validation_dogs_dir):
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    for bread_image in os.listdir(train_bread_dir):
        bread_image_path = os.path.join(train_bread_dir, bread_image)
        img = load_image(bread_image_path)
        if img is not None:
            train_images.append(img)
            train_labels.append(0)  

    for dog_image in os.listdir(train_dogs_dir):
        dog_image_path = os.path.join(train_dogs_dir, dog_image)
        img = load_image(dog_image_path)
        if img is not None:
            train_images.append(img)
            train_labels.append(1)  

    for bread_image in os.listdir(validation_bread_dir):
        bread_image_path = os.path.join(validation_bread_dir, bread_image)
        img = load_image(bread_image_path)
        if img is not None:
            validation_images.append(img)
            validation_labels.append(0)  

    for dog_image in os.listdir(validation_dogs_dir):
        dog_image_path = os.path.join(validation_dogs_dir, dog_image)
        img = load_image(dog_image_path)
        if img is not None:
            validation_images.append(img)
            validation_labels.append(1)  

    # 转换为NumPy数组
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)
    print(114514)
    return train_images,train_labels,validation_images, validation_labels