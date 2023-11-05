# 基于cnn网络的狗狗与面包识别（包含数据集）

## 下载数据集
使用爬虫下载数据集，对应python文件（Dataset_download.py）对应代码如下
```python
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-02 08:58:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 10:58:49
FilePath: \newrgzn\dataset_download.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import requests
import time
import os
import json.decoder  # 导入 JSONDecodeError 异常
from tqdm import tqdm
def download(str_zh):
    # ... 其他代码 ...
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
    }
    keyword = str_zh# 关键字
    max_page =30
    folder_path = './download/'+keyword
    os.makedirs(folder_path)
    i=1 # 记录图片数
    for page in tqdm(range(1, max_page + 1), desc="Processing Pages"):
        page = page * 30
        # 网址
        url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=' \
            + keyword + '&cl=2&lm=-&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word=' \
            + keyword + '&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=wallpaper&pn=' \
            + str(page) + '&rn=30&gsm=1e&1596899786625='
        #print(url)
        # 请求响应
        response = requests.get(url=url, headers=headers)
        try:
            # 得到相应的json数据
            json_data = response.json()
            if json_data.get('data'):
                for item in json_data.get('data')[:30]:
                    # 图片地址
                    img_url = item.get('thumbURL')
                    # 获取图片
                    image = requests.get(url=img_url)
                    # 下载图片

                    with open('./download/' + keyword + '/'+str(i)+'.jpg' , 'wb') as f:
                        f.write(image.content)  # 图片二进制数据
                    i += 1
        except json.decoder.JSONDecodeError as e:
            print(f"JSON解码错误: {e}")
            continue

    print('End!')
if __name__=="__main__":
    download("狗")
    download("面包")
  
```

## 数据集分类
需要将下载到的数据集按照55分成划分为训练集和验证集
并且生成对应目录文件（从download文件夹转为trans文件夹），并且将文件重新命名为标签+序号的格式的jpg文件，例如狗狗图片就命名为狗狗.count.jpg,方便作为标签进行训练
对应python文件为（）对应代码如下
```python
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-02 08:59:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 09:41:36
FilePath: \newrgzn\rename_files.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import random
import shutil

# 设置原始图片文件夹和目标文件夹
source_folder = 'download'
target_folder = 'trans'

# 创建目标文件夹和子文件夹
os.makedirs(os.path.join(target_folder, 'trains', 'bread'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'trains', 'dog'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'validation', 'bread'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'validation', 'dog'), exist_ok=True)

# 获取原始文件夹中的子文件夹（狗和面包）
subfolders = os.listdir(source_folder)

for subfolder in subfolders:
    source_subfolder = os.path.join(source_folder, subfolder)
    target_train_bread = os.path.join(target_folder, 'trains', 'bread')
    target_train_dog = os.path.join(target_folder, 'trains', 'dog')
    target_validation_bread = os.path.join(target_folder, 'validation', 'bread')
    target_validation_dog = os.path.join(target_folder, 'validation', 'dog')

    # 获取子文件夹中的所有图片文件
    image_files = os.listdir(source_subfolder)
    random.shuffle(image_files)

    # 计算前一半图片的数量
   
    print(len(image_files))
    print(len(image_files)//2)
    half_count = len(image_files) // 2

    # 将前一半图片移动到训练集文件夹
    for image_file in image_files[:half_count]:
        source_image_path = os.path.join(source_subfolder, image_file)
        if subfolder == '狗':
            target_image_path = os.path.join(target_train_dog, image_file)
        else:
            target_image_path = os.path.join(target_train_bread, image_file)
        shutil.move(source_image_path, target_image_path)

    # 将后一半图片移动到验证集文件夹
    for image_file in image_files[half_count:]:
        source_image_path = os.path.join(source_subfolder, image_file)
        if subfolder == '狗':
            target_image_path = os.path.join(target_validation_dog, image_file)
        else:
            target_image_path = os.path.join(target_validation_bread, image_file)
        shutil.move(source_image_path, target_image_path)

print("分割完成")

folder_paths = [
    'trans/trains/dog',
    'trans/trains/bread',
    'trans/validation/dog',
    'trans/validation/bread'
]

for folder_path in folder_paths:
    # 获取该文件夹下的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # 初始化序号
    counter = 1
    
    # 遍历图片文件并重命名
    for old_name in image_files:
        # 构建新的文件名
       
        new_name = f'{os.path.basename(folder_path)}.{counter}.jpg'
        # 构建旧文件的完整路径和新文件的完整路径
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_path, new_path)
        
        # 增加序号
        counter += 1

print('重命名完成')
  
```

## 数据集重命名
简历神经网络模型，通过函数getmodel（）可以获取模型（对应文件为Rename_files.py），对应代码如下
```python
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-15 08:41:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-04 15:36:29
FilePath: \newrgzn\my_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import tensorflow as tf


def getModel ():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])  
    model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model3 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model4= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    return model4
```
最后选用的是第四个模型


## 构建模型，导入数据集
```python
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






```
## 数据集清洗
```python

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
```
## 开始训练，保存模型
```python
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-16 14:22:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 10:34:00
FilePath: \newrgzn\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

'''
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑     永不宕机     永无BUG
'''

import tensorflow as tf
from cnn_trainer import cnn_train
# 获取可见的 GPU 设备列表
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

if gpu_devices:
    print("可用的 GPU 设备：")
    for device in gpu_devices:
        print(device.name)
    print("TensorFlow 正在使用 GPU 加速")
else:
    print("没有可用的 GPU 设备，TensorFlow 正在使用 CPU")
if __name__=="__main__":

    cnn_train()

```
## 模型的集成测试
```python
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


```
##  模型的自定义测试（基于pyqt5）
```python
import os
import cv2
from tensorflow.keras.optimizers import RMSprop

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import  image
from my_model import getModel

import tensorflow as tf

model =getModel()

model=tf.keras.models.load_model("./model/bread-dog.model")
path='mbg.png'
import numpy as np
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets




class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(959, 539)
        Dialog.setMinimumSize(QtCore.QSize(959, 539))
        Dialog.setMaximumSize(QtCore.QSize(19200, 10800))
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(0, 0, 959, 539))
        self.label.setAcceptDrops(False)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "blob"))
        self.label.setText(_translate("Dialog", "拖拽图片到此处"))


# 读取中文路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


class mwindow(QWidget, Ui_Dialog):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        # 调用Drops方法
        self.setAcceptDrops(True)
        # 图片
        self.img = None

    # 鼠标拖入事件
    def dragEnterEvent(self, evn):
        # print('鼠标拖入窗口了')
        # 鼠标放开函数事件
        evn.accept()

    # 鼠标放开执行
    def dropEvent(self, evn):
        # 判断文件类型
        if evn.mimeData().hasUrls():
            # 获取文件路径
            file_path = evn.mimeData().urls()[0].toLocalFile()
            # 判断是否是图片
            path=file_path
            img = cv2.imread(path)
            img = cv2.resize(img, (150, 150))

            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            classes = model.predict(images, batch_size=10)
            print(classes[0])



            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                self.img = cv_imread(file_path)
            else:
                # 提示
                QMessageBox.information(self, '提示', '请拖入图片')
                return
            # opencv 转qimage
            qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * self.img.shape[2],
                          QImage.Format_RGB888)
            # 显示图片 自适应
            self.label.setPixmap(
                QPixmap.fromImage(qimg).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))
        if classes[0] > 0:
            QMessageBox.information(self, '提示', '这是一只狗狗')
        else:
            QMessageBox.information(self, '提示', '这是一个面包')
        # print('鼠标放开了')

    def dragMoveEvent(self, evn):
        pass
        # print('鼠标移入')

import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 初始化窗口
    m = mwindow()
    m.show()
    sys.exit(app.exec_())




```
