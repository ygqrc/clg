'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-15 08:31:58
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-04 15:45:02
FilePath: \newrgzn\cnn_trainer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import sys
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import  image
from my_model import getModel
import tensorflow as tf
from ImageClassifier import getImageClassifiler
from classify import classifyImg
from test_orderator import getpath
from Data_Cleaner import clean_data
def cnn_train():
    print(sys.path)
    model =getModel()#读取模型
    model.summary()#打印模型
    #使用 RMSprop 优化器，学习率为0.001，使用二元交叉熵作为损失函数，并监测精度 ('acc') 作为性能指标。
    model.compile(optimizer=RMSprop(lr=0.001145141919810),
                loss='binary_crossentropy',
                metrics = ['acc'])

    #图像处理
    train_dir, validation_dir, train_bread_dir, train_dogs_dir, validation_bread_dir, validation_dogs_dir = getImageClassifiler()

    #标准化到[0,1]
    #创建用于训练和测试数据的图像数据生成器 (ImageDataGenerator)
    train_datagen = ImageDataGenerator( rescale = 1.0/255. )
    test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
    #加载图片集，数据生成器
    # 批量生成20个大小为150x150的图像及其标签用于训练
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    # 批量生成20个大小为150x150的图像及其标签用于验证
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )
    train_images,train_labels,validation_images, validation_labels=clean_data(train_bread_dir,train_dogs_dir,validation_bread_dir,validation_dogs_dir)

    history = model.fit(
        train_images, train_labels,
        epochs=20,
        validation_data=(validation_images, validation_labels),
        verbose=2
    )


    model.save("model/bread-dog-order.model")
    getpath(model)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(train_loss) + 1)
    # 绘制训练和验证损失
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证精度
    plt.figure()
    plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
#保存模型

    print("ok")
#本地验证





