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



