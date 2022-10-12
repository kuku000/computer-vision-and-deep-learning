import PyQt5
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QFileDialog, QLineEdit, QMessageBox
import cv2


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('2022 opencv HW1')
        self.setGeometry(50, 50, 900, 500)

        #background text
        self.mylabel1 = QLabel('1.Image Processing', self)
        self.mylabel1.move(450, 50)
        self.mylabel1.setFont(QFont('Times New Roman', 14))

        self.mylabel2 = QLabel('2.Image Smoothing', self)
        self.mylabel2.move(690, 50)
        self.mylabel2.setFont(QFont('Times New Roman', 14))

        self.loadlabel1 = QLabel('None', self)
        self.loadlabel1.setFont(QFont('Times New Roman', 8))
        self.loadlabel1.setGeometry(10,150,500,20)

        self.loadlabel2 = QLabel('None', self)
        self.loadlabel2.setFont(QFont('Times New Roman', 8))
        self.loadlabel2.setGeometry(10, 250, 500, 20)



        #buttom
        self.loadbuttom1 = QPushButton('Load Image 1', self)
        self.loadbuttom1.setGeometry(10, 100, 400, 40)
        self.loadbuttom1.clicked.connect(self.load_img)

        self.loadbuttom2 = QPushButton('Load Image 2', self)
        self.loadbuttom2.setGeometry(10, 200, 400, 40)
        self.loadbuttom2.clicked.connect(self.load_img2)

        self.mybutton1_1 = QPushButton('1-1 Color Separation', self)
        self.mybutton1_1.setGeometry(450, 100, 200, 40)
        self.mybutton1_1.clicked.connect(self.Color_Separation)

        self.mybutton1_2 = QPushButton('1-2 Color Transformation', self)
        self.mybutton1_2.setGeometry(450, 200, 200, 40)
        self.mybutton1_2.clicked.connect(self.Color_Transformation)

        self.mybutton1_3 = QPushButton('1-3 Color Detection', self)
        self.mybutton1_3.setGeometry(450, 300, 200, 40)
        self.mybutton1_3.clicked.connect(self.Color_Detection)

        self.mybutton1_4 = QPushButton('1-4 Blending', self)
        self.mybutton1_4.setGeometry(450, 400, 200, 40)
        self.mybutton1_4.clicked.connect(self.Blending)

        self.mybutton2_1 = QPushButton('2-1 Gaussian Blur ', self)
        self.mybutton2_1.setGeometry(690, 100, 200, 40)
        self.mybutton2_1.clicked.connect(self.gaussian)

        self.mybutton2_2 = QPushButton('2-2 Bilateral Filter ', self)
        self.mybutton2_2.setGeometry(690, 250, 200, 40)
        self.mybutton2_2.clicked.connect(self.bilateral)

        self.mybutton2_3 = QPushButton('2-2 Median Filter ', self)
        self.mybutton2_3.setGeometry(690, 400, 200, 40)
        self.mybutton2_3.clicked.connect(self.median)

    def load_img(self):
        filename, filterType = QFileDialog.getOpenFileNames(self,
                filter='JPEG (*.jpg);;PNG (*.png);;ALL File (*)')  # 選擇檔案對話視窗

        if filename:
            print(filename)
            print(filterType)
            self.img1 = cv2.imread(filename[0])
            cv2.imshow("picture select",self.img1)
            self.loadlabel1.setText(filename[0])

    def load_img2(self):
        filename, filterType = QFileDialog.getOpenFileNames(self,
                filter='JPEG (*.jpg);;PNG (*.png);;ALL File (*)')  # 選擇檔案對話視窗

        if filename:
            print(filename)
            print(filterType)
            self.img2 = cv2.imread(filename[0])
            cv2.imshow("picture select", self.img2)
            self.loadlabel2.setText(filename[0])

    def Color_Separation(self):
        try:
            Color_Separation(self.img1)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')

    def Color_Transformation(self):
        try:
            Color_Transformation(self.img1)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')


    def Color_Detection(self):
        try:
            Color_Detection(self.img1)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')

    def Blending(self):
        try:
            cv2.namedWindow("Blending")
            output = cv2.addWeighted(self.img1, 1, self.img2, 0, 50)
            cv2.imshow("Blending", output)
            cv2.createTrackbar('Blend', 'Blending', 0, 255, self.Alpha)  # 加入調整滑桿
            cv2.setTrackbarPos('Blend', 'Blending', 0)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')

    def Alpha(self,val):
        alpha = 1 - val / 255
        self.adjust(alpha)

    def adjust(self, alpha):
        output = cv2.addWeighted(self.img1, alpha, self.img2, 1 - alpha, 50)
        cv2.imshow("Blending", output)

    def gaussian(self):
        try:
            x = Gaussian_Filter(self.img1)
            x.gaussian_blur()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')

    def bilateral(self):
        try:
            x = Bilateral_Filter(self.img1)
            x.bilateral_filter()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')


    def median(self):
        try:
            x = Median_Filter(self.img1)
            x.median_filter()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'no image detected')



def Color_Separation(img):
    (B, G, R) = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype = np.uint8)

    R_img = cv2.merge([R, zeros, zeros])
    G_img = cv2.merge([zeros, G, zeros])
    B_img = cv2.merge([zeros, zeros, G])
    #cv2.imshow("R",cv2.merge([R, zeros, zeros]))
    #cv2.imshow("G",cv2.merge([zeros, G, zeros]))
    #cv2.imshow("B",cv2.merge([zeros, zeros, G]))
    cv2.imshow("RGB", np.hstack([R_img, G_img, B_img]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Color_Transformation(img):
    fnc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    A_W_img = img[:,:,0]/3 + img[:,:,1]/3 + img[:,:,2]/3
    A_W_img = np.array(A_W_img, dtype= np.uint8)

    cv2.imshow("RGB", fnc_img)
    cv2.imshow("",A_W_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def Color_Detection(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound_g = np.array([40, 50, 20])
    upper_bound_g = np.array([80, 255, 255])
    mask_g = cv2.inRange(hsv_img, lower_bound_g, upper_bound_g)
    output_g = cv2.bitwise_and(img, img, mask=mask_g)
    cv2.imshow("Green",output_g)

    lower_bound_w = np.array([0, 0, 20])
    upper_bound_w = np.array([180, 20, 255])
    mask_w = cv2.inRange(hsv_img, lower_bound_w, upper_bound_w)
    output_w = cv2.bitwise_and(img, img, mask=mask_w)
    cv2.imshow("White", output_w)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



class Filter:
    def __init__(self, img):
        self.img1 = img

    def kernal(self, val):
        size = 2*val + 1
        self.adjust(size)

    def adjust(self, size):
        output = cv2.GaussianBlur(self.img1, (size,size), 0)
        cv2.imshow("Gaussian Blur", output)

class Gaussian_Filter(Filter):

    def gaussian_blur(self):
        cv2.namedWindow("Gaussian Blur")
        cv2.imshow("Gaussian Blur", self.img1)
        cv2.createTrackbar('magnitube', 'Gaussian Blur', 0, 10, self.kernal)  # 加入調整滑桿
        cv2.setTrackbarPos('magnitube', 'Gaussian Blur', 0)






class Bilateral_Filter(Filter):

    def bilateral_filter(self):
        cv2.namedWindow("Bilateral Filter")
        cv2.imshow("Bilateral Filter", self.img1)
        cv2.createTrackbar('magnitube', 'Bilateral Filter', 0, 10, self.kernal)  # 加入調整滑桿
        cv2.setTrackbarPos('magnitube', 'Bilateral Filter', 0)


    def adjust(self, size):
        output = cv2.bilateralFilter(self.img1,size,90,90)
        cv2.imshow("Bilateral Filter", output)


class Median_Filter(Filter):

    def median_filter(self):
        cv2.namedWindow("Median Filter")
        cv2.imshow("Median Filter", self.img1)
        cv2.createTrackbar('magnitube', 'Median Filter', 0, 10, self.kernal)  # 加入調整滑桿
        cv2.setTrackbarPos('magnitube', 'Median Filter', 0)

    def adjust(self, size):
        output = cv2.medianBlur(self.img1, size)
        cv2.imshow("Median Filter", output)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())