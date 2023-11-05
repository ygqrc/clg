'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-15 08:45:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-04 10:47:24
FilePath: \newrgzn\ImageClassifier.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os

def getImageClassifiler():
    base_dir = 'trans'

    train_dir = os.path.join(base_dir, 'trains')
    validation_dir = os.path.join(base_dir, 'validation')

    train_bread_dir = os.path.join(train_dir, 'bread')
    train_dog_dir = os.path.join(train_dir, 'dog')

    validation_bread_dir = os.path.join(validation_dir, 'bread')
    validation_dog_dir = os.path.join(validation_dir, 'dog')

    # 返回一个元组，用括号明确表示
    return (train_dir, validation_dir, train_bread_dir, train_dog_dir, validation_bread_dir, validation_dog_dir)

def showImageClassifiler(train_bread_dir, train_dog_dir, validation_bread_dir, validation_dog_dir):
    print('total training bread images :', len(os.listdir(train_bread_dir)))
    print('total training dog images :', len(os.listdir(train_dog_dir)))

    print('total validation bread images :', len(os.listdir(validation_bread_dir)))
    print('total validation dog images :', len(os.listdir(validation_dog_dir)))
