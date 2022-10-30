import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import cv2
import random


from keras.applications.vgg19 import VGG19
from keras.layers import Dense,Flatten,Input,AvgPool2D,ReLU,Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras import optimizers
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy
from keras.models import load_model

from torchvision import models, transforms, datasets
from torchsummary import summary



from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QFileDialog, QLineEdit, QMessageBox
import PyQt5.QtCore as core
import sys

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.img = None
        self.img2 = None

    def initUI(self):
        self.setWindowTitle('2022 opencv HW1-5')
        self.setGeometry(50, 50, 900, 500)

        #layout = QGridLayout()
        #self.setLayout(layout)

        self.piclabel = QLabel(self)
        self.piclabel.setGeometry(500,100,400,400)
        #layout.addWidget(self.piclabel, 0, 0, 1, 2)

        self.loadlabel1 = QLabel('None', self)
        self.loadlabel1.setFont(QFont('Times New Roman', 8))
        self.loadlabel1.setGeometry(10,100,400,20)

        self.loadlabel2 = QLabel(self)
        self.loadlabel2.setFont(QFont('Times New Roman', 12))
        self.loadlabel2.setGeometry(500, 10, 200, 20)

        self.loadlabel3 = QLabel(self)
        self.loadlabel3.setFont(QFont('Times New Roman', 12))
        self.loadlabel3.setGeometry(500, 30, 200, 20)




        #buttom
        self.loadbuttom1 = QPushButton('Load Image', self)
        self.loadbuttom1.setGeometry(10, 50, 300, 40)
        self.loadbuttom1.clicked.connect(self.load_img)


        self.mybutton1_1 = QPushButton('1.show Train Images', self)
        self.mybutton1_1.setGeometry(10, 150, 300, 40)
        self.mybutton1_1.clicked.connect(self.show_Train_Images_connect)

        self.mybutton1_2 = QPushButton('2.Show Model Structure', self)
        self.mybutton1_2.setGeometry(10, 200, 300, 40)
        self.mybutton1_2.clicked.connect(self.Show_Model_Structure_connect)

        self.mybutton1_3 = QPushButton('3.Show Data Augmentation', self)
        self.mybutton1_3.setGeometry(10, 250, 300, 40)
        self.mybutton1_3.clicked.connect(self.Show_Data_Augmentation_connect)

        self.mybutton1_4 = QPushButton('4.Show Accuracy and loss', self)
        self.mybutton1_4.setGeometry(10, 300, 300, 40)
        self.mybutton1_4.clicked.connect(self.Show_Accuracy_and_loss_connect)

        self.mybutton1_5 = QPushButton('5.Inference', self)
        self.mybutton1_5.setGeometry(10, 350, 300, 40)
        self.mybutton1_5.clicked.connect(self.infer_connect)



    def show_Train_Images_connect(self):
        try:
            show_9_pic()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def Show_Model_Structure_connect(self):
        try:
            show_vgg_19()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def Show_Data_Augmentation_connect(self):
        try:
            print(self.img2)
            data_Augmentation(self.img2)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def Show_Accuracy_and_loss_connect(self):
        try:
            loss_img = plt.imread('./pictures/acc_loss.png')
            plt.figure(figsize=(40,40))
            plt.imshow(loss_img)
            plt.show()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def infer_connect(self):
        model = load_model('model_v1.h5')
        print(model.summary())
        print(self.img.shape)
        pre_y = model.predict(self.img.reshape((1,32,32,3)))
        label = []
        for i in range(10):
            label.append(pre_y[0][i])
        print(label)
        confid = (max(label)/sum(label)).round(2)
        print(confid)
        pre_label = label.index(max(label))
        print(pre_label)
        if pre_label == 0:
            ans = 'airplane'
        elif pre_label == 1:
            ans = 'automobile'
        elif pre_label == 2:
            ans = 'bird'
        elif pre_label == 3:
            ans = 'cat'
        elif pre_label == 4:
            ans = 'deer'
        elif pre_label == 5:
            ans = 'dog'
        elif pre_label == 6:
            ans = 'frog'
        elif pre_label == 7:
            ans = 'horse'
        elif pre_label == 8:
            ans = 'ship'
        elif pre_label == 9:
            ans = 'truck'
        self.loadlabel2.setText('Confidence = ' + str(confid))
        self.loadlabel3.setText('Predict label: ' + str(ans))

    def load_img(self):
        filename, filterType = QFileDialog.getOpenFileNames(self,
                filter='JPEG (*.jpg);;PNG (*.png);;ALL File (*)')  # 選擇檔案對話視窗

        if filename:
            print(filename[0])
            print(filterType)
            self.img = cv2.imread(filename[0])
            self.img = cv2.resize(self.img, (32, 32))
            self.img2 = PIL.Image.open(filename[0])
            cv2.imshow("picture select", self.img)
            self.loadlabel1.setText(filename[0])
            self.mypixmap = QPixmap(filename[0])
            self.mypixmap = self.mypixmap.scaled(400, 400, core.Qt.KeepAspectRatio)
            if self.mypixmap.isNull():
                print('load image failed')
                return
            self.piclabel.setPixmap(self.mypixmap)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Shape of x_train is ",x_train.shape)
print("Shape of y_train is ",y_train.shape)
print("Shape of x_test  is ",x_test.shape)
print("Shape of y_test  is",y_test.shape)
def show_9_pic():
    data = range(len(x_train))
    train_show = random.sample(data,9)
    print(train_show)
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        if y_train[train_show[i]] == 0:
            plt.title('airplane')
        elif y_train[train_show[i]] == 1:
            plt.title('automobile')
        elif y_train[train_show[i]] == 2:
            plt.title('bird')
        elif y_train[train_show[i]] == 3:
            plt.title('cat')
        elif y_train[train_show[i]] == 4:
            plt.title('deer')
        elif y_train[train_show[i]] == 5:
            plt.title('dog')
        elif y_train[train_show[i]] == 6:
            plt.title('frog')
        elif y_train[train_show[i]] == 7:
            plt.title('horse')
        elif y_train[train_show[i]] == 8:
            plt.title('ship')
        elif y_train[train_show[i]] == 9:
            plt.title('truck')
        plt.imshow(x_train[train_show[i],:,:,:])
    fig.tight_layout()
    plt.show()

def show_vgg_19(see = True):
    model = models.vgg19(num_classes = 10)
    x=summary(model, (3, 32, 32))
    if see:
        print(x)
    return model

show_vgg_19()

def data_Augmentation(img):
    img_list = []
    tranform = transforms.RandomHorizontalFlip(p=1)(img)
    transform1 = transforms.RandomGrayscale(p=1)(img)
    transform2 = transforms.RandomAffine(45)(img)
    img_list.append(np.array(tranform.getdata()).reshape(tranform.size[1], tranform.size[0], 3))
    img_list.append(np.array(transform1.getdata()).reshape(transform1.size[1], transform1.size[0], 3))
    img_list.append(np.array(transform2.getdata()).reshape(transform2.size[1], transform2.size[0], 3))
    print(type(tranform))
    fig = plt.figure()
    for i in range(3):
        #print(1)
        plt.subplot(1,3, i + 1)
        plt.imshow(img_list[i])
    fig.tight_layout()
    plt.show()

img = PIL.Image.open('./pictures/5.jpg')

#plt.show()
#data_Augmentation(img)


def training():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(y_train.shape)
    print(y_test.shape)
    model = VGG19(weights="imagenet", include_top = False, input_shape=(32, 32, 3))
    models = Sequential()
    for layer in model.layers:
        models.add(layer)
    for layer in models.layers:
        layer.trainable = False
    models.add(Flatten())
    models.add(Dense(4096))
    models.add(ReLU())
    models.add(Dropout(0.2))
    models.add(Dense(1000))
    models.add(ReLU())
    models.add(Dropout(0.2))
    models.add(Dense(10, activation="softmax"))
    print(models.summary())

    optimizer = optimizers.Adam(learning_rate=0.001)
    models.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = models.fit(x_train, y_train, epochs=30, batch_size = 8, validation_data = (x_test, y_test))
    models.save('model_v1.h5')
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./pictures/acc_loss.png')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())


#training()






