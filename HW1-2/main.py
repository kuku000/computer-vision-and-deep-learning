import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as maimg
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QFileDialog, QLineEdit, QMessageBox
import sys

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('2022 opencv HW1')
        self.setGeometry(50, 50, 900, 500)

        #background text
        self.mylabel1 = QLabel('3.Edge Detection', self)
        self.mylabel1.move(450, 50)
        self.mylabel1.setFont(QFont('Times New Roman', 14))

        self.mylabel2 = QLabel('4.Transformation', self)
        self.mylabel2.move(690, 50)
        self.mylabel2.setFont(QFont('Times New Roman', 14))

        self.loadlabel1 = QLabel('None', self)
        self.loadlabel1.setFont(QFont('Times New Roman', 8))
        self.loadlabel1.setGeometry(10,250,500,20)

        #buttom
        self.loadbuttom1 = QPushButton('Load Image', self)
        self.loadbuttom1.setGeometry(10, 200, 400, 40)
        self.loadbuttom1.clicked.connect(self.load_img)

        self.mybutton1_1 = QPushButton('3-1 Gaussian Blur', self)
        self.mybutton1_1.setGeometry(450, 100, 200, 40)
        self.mybutton1_1.clicked.connect(self.gaussian_connect)

        self.mybutton1_2 = QPushButton('3-2 Sobel X', self)
        self.mybutton1_2.setGeometry(450, 200, 200, 40)
        self.mybutton1_2.clicked.connect(self.sobel_X_connect)

        self.mybutton1_3 = QPushButton('3-3 Sobel Y', self)
        self.mybutton1_3.setGeometry(450, 300, 200, 40)
        self.mybutton1_3.clicked.connect(self.sobel_Y_connect)

        self.mybutton1_4 = QPushButton('3-4 Magnitude', self)
        self.mybutton1_4.setGeometry(450, 400, 200, 40)
        self.mybutton1_4.clicked.connect(self.magnitude_connect)

        self.mybutton2_1 = QPushButton('4-1 Resize', self)
        self.mybutton2_1.setGeometry(690, 100, 200, 40)
        self.mybutton2_1.clicked.connect(self.resize_connect)

        self.mybutton2_2 = QPushButton('4-2 Translation', self)
        self.mybutton2_2.setGeometry(690, 200, 200, 40)
        self.mybutton2_2.clicked.connect(self.translation_connect)

        self.mybutton2_3 = QPushButton('4-3 Rotation,scaling', self)
        self.mybutton2_3.setGeometry(690, 300, 200, 40)
        self.mybutton2_3.clicked.connect(self.rotation_connect)

        self.mybutton2_3 = QPushButton('4-4 Shearing', self)
        self.mybutton2_3.setGeometry(690, 400, 200, 40)
        self.mybutton2_3.clicked.connect(self.shearing_connect)

    def load_img(self):
        filename, filterType = QFileDialog.getOpenFileNames(self,
                filter='JPEG (*.jpg);;PNG (*.png);;ALL File (*)')  # 選擇檔案對話視窗

        if filename:
            print(filename)
            print(filterType)
            self.img = cv2.imread(filename[0])
            cv2.imshow("picture select", self.img)
            self.loadlabel1.setText(filename[0])


    #img load
    #img_1 = plt.imread('./Dataset_OpenCvDl_Hw1_2/Q3_Image/building.jpg')
    #img_2 = plt.imread('./Dataset_OpenCvDl_Hw1_2/Q4_Image/Microsoft.png')

    #connect------------------------------------------------------------
    def gaussian_connect(self):
        try:
            gray_scale_guassian(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self,'info','on image detected')

    def sobel_X_connect(self):
        try:
            gray_scale_Sobel_X(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'on image detected')

    def sobel_Y_connect(self):
        try:
            gray_scale_Sobel_Y(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'on image detected')

    def magnitude_connect(self):
        try:
            magnitude(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'on image detected')

    def resize_connect(self):
        try:
            resize()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def translation_connect(self):
        try:
            translation()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def rotation_connect(self):
        try:
            rotation()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def shearing_connect(self):
        try:
            shearing()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')


def gray_scale_guassian(rgb):
    rows = 2
    gray =np.dot(rgb[...,:3],[0.299, 0.587, 0.144])
    kernel = gaussian_kernel()
    print(type(kernel))
    img_gauss = conv(gray, kernel)
    fig = plt.figure(figsize= [40,20])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Grayscale')
    ax1.imshow(gray, cmap=plt.get_cmap('gray'))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Gaussian Blur')
    ax2.imshow(img_gauss, cmap=plt.get_cmap('gray'))
    plt.savefig('Gaussian_building.png')
    plt.show()

def gray_scale_Sobel_X(rgb,show = True):
    rows = 2
    gray =np.dot(rgb[...,:3],[0.299, 0.587, 0.144])
    kernel = gaussian_kernel()
    print(type(kernel))
    img_gauss = conv(gray, kernel)
    kernel_2=Sobel_X_kernel()
    img_sobel_x = conv(img_gauss, kernel_2)
    for i in range(img_sobel_x.shape[0]):
        for j in range(img_sobel_x.shape[1]):
            if img_sobel_x[i][j] <64:
                img_sobel_x[i][j] = 0
                print("抓到你摟")
            if img_sobel_x[i][j] >240:
                img_sobel_x[i][j] = 255
                print("抓到你摟")
    if show:
        fig = plt.figure()
        plt.imshow(img_sobel_x, cmap=plt.get_cmap('gray'))
        plt.savefig("Sobel_X_building.png")
        plt.show()
    return abs(img_sobel_x)

def gray_scale_Sobel_Y(rgb,show = True):
    rows = 2
    gray =np.dot(rgb[...,:3],[0.299, 0.587, 0.144])
    kernel = gaussian_kernel()
    print(type(kernel))
    img_gauss = conv(gray, kernel)
    kernel_2=Sobel_y_kernel()
    img_sobel_y = conv(img_gauss, kernel_2)
    for i in range(img_sobel_y.shape[0]):
        for j in range(img_sobel_y.shape[1]):
            if img_sobel_y[i][j] <64:
                img_sobel_y[i][j] = 0
                print("抓到你摟")
            if img_sobel_y[i][j] >240:
                img_sobel_y[i][j] = 255
                print("抓到你摟")

    if show:
        fig = plt.figure()
        plt.imshow(img_sobel_y, cmap=plt.get_cmap('gray'))
        plt.savefig("Sobel_Y_building.png")
        plt.show()
    return abs(img_sobel_y)

def gaussian_kernel():
    x, y = np.mgrid[-1:2, -1:2]
    print(x)
    print(y)
    kernel = np.exp(-(x ** 2 + y ** 2))
    kernel = kernel / kernel.sum()
    print(kernel)
    return kernel
def Sobel_X_kernel():
    kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
    return np.array(kernel)

def Sobel_y_kernel():
    kernel = [[1,2,1],[0,0,0],[-1,-2,-1]]
    return np.array(kernel)

def conv(img, kernel):
    k_size = kernel.shape[0]
    img_x, img_y = img.shape
    padding = np.pad(img, (k_size-2 , k_size-2 ),'edge')
    print(padding)
    conv_img = []
    for i in range(img_x):
        for j in range(img_y):
            conv_img.append(np.sum(padding[i:k_size + i,j:k_size + j] * kernel))
    #1D list
    print(np.size(conv_img))
    #1D->2D array using reshape
    conv_img = np.array(conv_img).reshape([img_x,img_y])
    return np.abs(conv_img)

def magnitude(img):
    X = gray_scale_Sobel_X(img,False)
    Y = gray_scale_Sobel_Y(img,False)

    img3 = abs(X**2 +Y**2)**(1/2)
    plt.figure()
    plt.savefig("Magnitude.png")
    plt.imshow(img3, cmap=plt.get_cmap('gray'))
    plt.show()

#------------------------------------------------------------------------------------------
#4-1 resize
def resize():
    img_2 = plt.imread('./Dataset_OpenCvDl_Hw1_2/Q4_Image/Microsoft.png')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    constant = cv2.copyMakeBorder(img_2, 0, 430, 0, 430, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_2 = cv2.resize(constant, (215, 215))
    cv2.imwrite("Resize.png", img_2*255)
    cv2.imshow('resize',img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#resize(img_2)
#4-2
def translation():
    img_2 = plt.imread('./Dataset_OpenCvDl_Hw1_2/Q4_Image/Microsoft.png')
    img =plt.imread("Resize.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    constant = cv2.copyMakeBorder(img_2, 430, 0,430, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_2 = cv2.resize(constant, (215, 215))
    img_trans = cv2.add(img,img_2)
    cv2.imwrite("Translation.png", img_trans * 255)
    cv2.imshow('resize', img_trans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#translation(img_2)

#4-3
def rotation():
    img =plt.imread("Translation.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    constant = cv2.copyMakeBorder(img, 108, 108, 108, 108, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_scaled = cv2.resize(constant, (215, 215))
    img_scaled = np.asarray(img_scaled, dtype=np.float32)
    M = cv2.getRotationMatrix2D( (108,108), 45, 1)
    size=(img_scaled.shape[0],img_scaled.shape[1])
    img_rota = cv2.warpAffine(img_scaled, M,dsize=size )
    cv2.imwrite("Rotation.png", img_rota * 255)
    cv2.imshow('Rotation', img_rota)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#4-4
def shearing():
    img = plt.imread("Rotation.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    M = np.float32([[1, 0.4, 1], [0, 1, 0]])
    M[0, 2] = -M[0, 1] * 215 / 2
    M[1, 2] = -M[1, 0] * 215 / 2
    print(M)
    img_shear = cv2.warpAffine(img, M, (215, 215))
    cv2.imwrite("Shearing.png", img_shear * 255)
    cv2.imshow('Shearing', img_shear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())
