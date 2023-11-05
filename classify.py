'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-15 08:58:46
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-02 10:35:52
FilePath: \newrgzn\classify.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2

def classifyImg(model):
    path=''
    import numpy as np
    import cv2

    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))

    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0:
     print("这是一只狗狗")
    else:
     print("这是一块面包")
def classify(img_path):
    # 灰度读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (150, 150))
        data = img.reshape(-1, 150, 150, 1)
        return data
