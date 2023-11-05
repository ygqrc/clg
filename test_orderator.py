'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-04 09:09:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-04 09:28:00
FilePath: \newrgzn\test_orderator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os

# 指定文件夹的路径

    
import numpy as np
import cv2
def re(path,model):
    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))

    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0:
     print(path+":狗狗")
    else:
     print(path+":面包")
# 列出文件夹下的所有文件路径
def getpath(model):
    folder_path = 'C:\\Users\\Administrator\\Desktop\\tested'
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    for path in file_paths:
        re(path,model)

