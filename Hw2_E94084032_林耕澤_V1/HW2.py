#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os 
import cv2
import numpy as np
from matplotlib import pyplot as plt

import PyQt5
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,\
QLabel, QGroupBox, QBoxLayout, QGridLayout, QFileDialog, QComboBox, QTextEdit,\
QMessageBox
from PyQt5.QtGui import QFont, QImage, QPixmap

import copy

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def group1(self):  
        group_box = QGroupBox("load")
        group_box.setFont(QFont('Times New Roman',12))
        
        loadf = QPushButton('load folder', self)
        loadf.clicked.connect(self.open_folder)
        self.mylabel1 = QLabel('None', self)
        
        load_L = QPushButton('load img_L', self)
        load_L.clicked.connect(self.openfile_connect_L)
        self.mylabel_L = QLabel('None', self)
        
        load_R = QPushButton('load img_R', self)
        load_R.clicked.connect(self.openfile_connect_R)
        self.mylabel_R = QLabel('None', self)
        
        qbox = QBoxLayout(2)
        
        qbox.addWidget(loadf)   
        qbox.addWidget(self.mylabel1)   
        qbox.addWidget(load_L)  
        qbox.addWidget(self.mylabel_L)
        qbox.addWidget(load_R)  
        qbox.addWidget(self.mylabel_R)
    
        qbox.addStretch(1)
        group_box.setLayout(qbox)
        
        return group_box
        
    def group2(self):        
        
        group_box = QGroupBox("1 find contour")
        group_box.setFont(QFont('Times New Roman',12))
        
        draw_contour = QPushButton("1.1 draw contour")
        draw_contour.clicked.connect(self.draw_contour_connect)
        
        count_rings = QPushButton("1.2 count rings")
        count_rings.clicked.connect(self.count_rings_connect)
        
        self.num1 = QLabel('There are_ rings in img1', self)
        self.num2 = QLabel('There are_ rings in img2', self)

        vbox = QVBoxLayout()
        
        vbox.addWidget(draw_contour)
        vbox.addWidget(count_rings)
        vbox.addWidget(self.num1)
        vbox.addWidget(self.num2)
        
        vbox.addStretch(1)
        group_box.setLayout(vbox)
        
        return group_box
    
    def group3(self):        
        
        group_box = QGroupBox("2 Calibration")
        group_box.setFont(QFont('Times New Roman',12))
        
        corner_detect = QPushButton("2.1 corner detection")
        corner_detect.clicked.connect(self.corner_detect_connect)
        
        show_intrinsic_matrix = QPushButton("2.2 find intrinsic")
        show_intrinsic_matrix.clicked.connect(self.show_intrinsic_matrix_connect)
        
        extrinsic_choose_box = QGroupBox("2.3 find extrinsic")
        
        show_extrinsic_matrix = QPushButton("2.3 find extrinsic")
        show_extrinsic_matrix.clicked.connect(self.show_extrinsic_matrix_connect)
        
        
        self.mycombobox = QComboBox(self)
        self.mycombobox.addItems(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        
        qvbox0 = QVBoxLayout()
        qvbox0.addWidget(self.mycombobox)
        qvbox0.addWidget(show_extrinsic_matrix)
        
        extrinsic_choose_box.setLayout(qvbox0)
        
        show_distortion_matrix = QPushButton("2.4 find the distortion matrix")
        show_distortion_matrix.clicked.connect(self.show_distortion_matrix_connect)
        
        undistorted = QPushButton("2.5 show result")
        undistorted.clicked.connect(self.undistorted_connect)
        

        qvbox = QVBoxLayout()
        
        qvbox.addWidget(corner_detect)
        qvbox.addWidget(show_intrinsic_matrix)
        qvbox.addWidget(extrinsic_choose_box)
        qvbox.addWidget(show_distortion_matrix)
        qvbox.addWidget(undistorted)

        qvbox.addStretch(1)
        group_box.setLayout(qvbox)
        
        return group_box
    
    def group4(self):  
        
        group_box = QGroupBox("3 rugmented reality")
        group_box.setFont(QFont('Times New Roman',12))
        
        self.text = QTextEdit()
       
        show_words = QPushButton("3.1 show words on board")
        show_words.clicked.connect(self.show_words_connect)
        
        show_words_vertically = QPushButton("3.2 show words vertically")
        show_words_vertically.clicked.connect(self.show_words_vertically_connect)
        
        qvbox = QVBoxLayout()
        
        qvbox.addWidget(self.text)
        qvbox.addWidget(show_words)
        qvbox.addWidget(show_words_vertically)


        
        qvbox.addStretch(1)
        group_box.setLayout(qvbox)
        
        return group_box
        
    def group5(self):        
        
        group_box = QGroupBox("4 stereo disparity map")
        group_box.setFont(QFont('Times New Roman',12))
        
        
        disparity_map = QPushButton("4.1 corner detection")
        disparity_map.clicked.connect(self.disparity_map_connect)

        qvbox = QVBoxLayout()
        qvbox.addWidget(disparity_map)
        qvbox.addStretch(1)
        group_box.setLayout(qvbox)
        
        return group_box


    def initUI(self):
        self.setWindowTitle('opencv_E94084032_HW2')
        self.setGeometry(100, 100, 1000, 180)
        
        grid = QGridLayout()

        grid.addWidget(self.group1(),0,0,1,4)       
        grid.addWidget(self.group2(),1,0)
        grid.addWidget(self.group3(),1,1)
        grid.addWidget(self.group4(),1,2)
        grid.addWidget(self.group5(),1,3)
        self.setLayout(grid)
        
    def load_images(self,folder):
        try:
            images = []
            for filename in os.listdir(folder) :
                if filename.endswith(".bmp"):
                    img = cv2.imread(os.path.join(folder,filename))
                    if img is not None:
                        images.append(img)
            return images
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')

    def open_folder(self):
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
            self.img = self.load_images(folder_path)
            x=QFileInfo(folder_path)
            self.mylabel1.setText(x.fileName())
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
    
    def openfile_connect_L(self):
        try:
            filename, filetype = QFileDialog.getOpenFileName(self,"choose")
            if (filename!=""):
                filename1=filename
            x=QFileInfo(filename1)
            self.mylabel_L.setText(x.fileName())
            self.img_L = cv2.imread(filename1)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
                 
    def openfile_connect_R(self):
        try:
            filename, filetype = QFileDialog.getOpenFileName(self,"choose")
            if (filename!=""):
                filename1=filename
            x=QFileInfo(filename1)
            self.mylabel_R.setText(x.fileName())
            self.img_R = cv2.imread(filename1)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def draw_contour_connect(self):
        try:
            if __name__ == '__main__':
                a=EX1()
                a.draw_contour()
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def count_rings_connect(self):
        try:
            if __name__ == '__main__':
                a=EX1()
                n1,n2 = a.count_rings()
                self.num1.setText('There are {} rings in img1'.format(n1))
                self.num2.setText('There are {} rings in img2'.format(n2))
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
    
    def corner_detect_connect(self):
        try:
            if __name__ == '__main__':
                a=EX2()
                a.corner_detect(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def show_intrinsic_matrix_connect(self):
        try:
            if __name__ == '__main__':
                a=EX2()
                a.show_intrinsic_matrix(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def show_extrinsic_matrix_connect(self):
        try:
            if __name__ == '__main__':
                a=EX2()
                a.show_extrinsic_matrix(self.mycombobox.currentIndex(),self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
    
    def show_distortion_matrix_connect(self):
        try:
            if __name__ == '__main__':
                a=EX2()
                a.show_distortion_matrix(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def undistorted_connect(self):
        try:
            if __name__ == '__main__':
                a=EX2()
                a.undistorted(self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def show_words_connect(self):
        try:
            if __name__ == '__main__':
                a=EX3()
                a.show_words(self.text.toPlainText(),self.img)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def show_words_vertically_connect(self):
        try:
            if __name__ == '__main__':
                a=EX3()
                a.show_words_vertically(self.text.toPlainText(),self.img) 
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            
    def disparity_map_connect(self):
        try:
            if __name__ == '__main__':
                a=EX4()
                a.disparity_map(self.img_L,self.img_R)
        except:
            mbox = QMessageBox(self)
            mbox.information(self, 'info', 'Error')
            

class EX1:
    def draw_contour(self):
        img = cv2.imread("./Dataset_OpenCvDl_Hw2/Q1_Image/img1.jpg")                         
        img2 = cv2.imread("./Dataset_OpenCvDl_Hw2/Q1_Image/img2.jpg") 
        
        resize_img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)
        resize_img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(resize_img2, cv2.COLOR_BGR2GRAY)
        
        blurr = cv2.GaussianBlur(gray, (11, 11), 0)
        blurr2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        
        canny = cv2.Canny(blurr, 127, 255)
        canny2 = cv2.Canny(blurr2, 127, 255)
        
        edge_image, contour, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_image, contour2, hierarchy = cv2.findContours(canny2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        

        contour_ans=cv2.drawContours(resize_img, contour, -1, (180, 0, 120), 2)
        contour_ans2=cv2.drawContours(resize_img2, contour2, -1, (180, 0, 120), 2)

        cv2.imshow('draw contour', contour_ans)
        cv2.imshow('draw contour2', contour_ans2)
        cv2.waitKey(0)
        cv2.destroyWindow('draw contour')
        cv2.destroyWindow('draw contour2')
        
    def count_rings(self):
        img = cv2.imread("./Dataset_OpenCvDl_Hw2/Q1_Image/img1.jpg")
        img2 = cv2.cv2.imread("./Dataset_OpenCvDl_Hw2/Q1_Image/img2.jpg")
        
        resize_img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY )
        blurr = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blurr, 127, 255)
        edge_image, contour, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        resize_img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2), interpolation=cv2.INTER_AREA)
        gray2 = cv2.cvtColor(resize_img2, cv2.COLOR_BGR2GRAY )
        blurr2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        canny2 = cv2.Canny(blurr2, 127, 255)
        edge_image2, contour2, hierarchy2 = cv2.findContours(canny2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count1, count2=len(contour), len(contour2)
        
        return count1, count2
    
    
class EX2:
    def corner_detect(self,img):
        nx = 12
        ny = 9
        board_img = copy.deepcopy(img)
        for i in range(len(board_img)):
            ret, cp_img = cv2.findChessboardCorners( board_img[i], (nx-1,ny-1), None)
            cv2.drawChessboardCorners(board_img[i], (nx-1,ny-1), cp_img, ret)
            cv2.namedWindow("board",cv2.WINDOW_NORMAL)
            cv2.resizeWindow('board', 512, 512)
            cv2.imshow("board", board_img[i])
            cv2.waitKey(500)
        cv2.waitKey(0)
        cv2.cv2.destroyWindow('board')
        
    def mtx_dist_rvecs_tvecs(self,img):
        nx = 12
        ny = 9
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        objpoints = []  
        imgpoints = []  
        for i in range(0,len(img)):
            image = img[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx-1,ny-1), None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image.shape[1], image.shape[0]), None, None)
        
        
        return ret, mtx, dist, rvecs, tvecs
        
        

    def show_intrinsic_matrix(self,img):
        ret, mtx, dist, rvecs, tvecs = self.mtx_dist_rvecs_tvecs(img)
    
        print(mtx)  
        
    def show_extrinsic_matrix(self,num,img):
        ret, mtx, dist, rvecs, tvecs = self.mtx_dist_rvecs_tvecs(img)
        R = cv2.Rodrigues(rvecs[num-1])
        ext = np.hstack((R[0], tvecs[num-1]))
        print(ext)
        
    def show_distortion_matrix(self,img):
        ret, mtx, dist, rvecs, tvecs = self.mtx_dist_rvecs_tvecs(img)
        print(dist)
        
    def undistorted(self,img):
        undistorted_img = copy.deepcopy(img)
        
        nx = 12
        ny = 9
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []  
        imgpoints = []  
        for i in range(0,len(img)):
            image = img[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx-1,ny-1), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
        ret, EX2.mtx, EX2.dist, EX2.rvecs, EX2.tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image.shape[1], image.shape[0]), None, None)
        
        for i in range(0,len(img)):
            undist = cv2.undistort(undistorted_img[i],EX2.mtx,EX2.dist, None) 
            imgs = np.hstack([img[i],undist])
            cv2.namedWindow("undistortion",cv2.WINDOW_NORMAL)
            cv2.resizeWindow('undistortion', 1024, 512)
            cv2.imshow("undistortion",imgs)
            cv2.waitKey(500)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class EX3:
    def show_words(self,words,img):
        nx = 12
        ny = 9
        self.cache = copy.deepcopy(img)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []  
        imgpoints = []  
        for i in range(0,5):
            image = self.cache[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx-1,ny-1), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (self.cache[0].shape[1], self.cache[0].shape[0]), None, None)
        fs = cv2.FileStorage('./Dataset_OpenCvDl_Hw2/Q3_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        ch = []
        for i in range(0,len(words)):
            if i>5:
                break
            ch.append(fs.getNode(words[i]).mat())
            if i==0:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[7,5,0]
                    ch[i][k][1]+=[7,5,0]
            elif i==1:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[4,5,0]
                    ch[i][k][1]+=[4,5,0]
            elif i==2:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[1,5,0]
                    ch[i][k][1]+=[1,5,0]
            elif i==3:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[7,2,0]
                    ch[i][k][1]+=[7,2,0]
            elif i==4:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[4,2,0]
                    ch[i][k][1]+=[4,2,0]
            else:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0]+=[1,2,0]
                    ch[i][k][1]+=[1,2,0]
        for i in range(0,5):
            for j in range(0,len(words)):
                if j>5:
                    break
                for k in range(0,ch[j].shape[0]):
                    imgpts, jac = cv2.projectPoints(np.float32(ch[j][k]).reshape(-1, 3), rvecs[i], tvecs[i], mtx, dist)
                    self.cache[i] = cv2.line(self.cache[i], tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()),(100, 0, 255), 5)
            
            self.cache[i] = cv2.resize(self.cache[i], (int(self.cache[i].shape[1]/4), int(self.cache[i].shape[0]/4)), interpolation=cv2.INTER_AREA)
            cv2.imshow('show words on board',self.cache[i])
            cv2.waitKey(500)
        cv2.waitKey(0)
        cv2.destroyWindow('show words on board')

    def show_words_vertically(self,words,img):
        nx = 12
        ny =9
        self.cache = copy.deepcopy(img)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []  
        imgpoints = []  
        for i in range(0,5):
            image = self.cache[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx-1,ny-1), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)     
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (self.cache[0].shape[1], self.cache[0].shape[0]), None, None)
        fs = cv2.FileStorage('./Dataset_OpenCvDl_Hw2/Q3_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        ch = []
        for i in range(0,len(words)):
            if i > 5:
                break
            ch.append(fs.getNode(words[i]).mat())
            if i == 0:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [7,5,0]
                    ch[i][k][1] += [7,5,0]
            elif i == 1:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [4,5,0]
                    ch[i][k][1] += [4,5,0]
            elif i == 2:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [1,5,0]
                    ch[i][k][1] += [1,5,0]
            elif i == 3:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [7,2,0]
                    ch[i][k][1] += [7,2,0]
            elif i == 4:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [4,2,0]
                    ch[i][k][1] += [4,2,0]
            else:
                for k in range(0,ch[i].shape[0]):
                    ch[i][k][0] += [1,2,0]
                    ch[i][k][1] += [1,2,0]    
        for i in range(0,5):
            for j in range(0,len(words)):
                if j>5:
                    break
                for k in range(0,ch[j].shape[0]):
                    imgpts, jac = cv2.projectPoints(np.float32(ch[j][k]).reshape(-1, 3), rvecs[i], tvecs[i], mtx, dist)
                    self.cache[i] = cv2.line(self.cache[i], tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (100, 0, 255), 5)
            self.cache[i] = cv2.resize(self.cache[i], (int(self.cache[i].shape[1]/4), int(self.cache[i].shape[0]/4)), interpolation=cv2.INTER_AREA)
            cv2.imshow("show words vertically",self.cache[i])
            cv2.waitKey(500)
        cv2.waitKey(0)
        cv2.destroyWindow('show words vertically')

class EX4:
    def disparity_map(self,img_L,img_R):
        
        img_L_g = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        img_R_g = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        
        if img_L.shape[0]>img_L.shape[1] :
            multiple= img_L.shape[0]/255
        else:
            multiple= img_L.shape[1]/255
            
        x, y = int(img_L.shape[1]/multiple),int(img_L.shape[0]/multiple)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(img_L_g,img_R_g).astype(np.float32)/16 
        
        norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        self.imL = cv2.resize(img_L, (x,y))  
        self.imR = cv2.resize(img_R, (x,y))  
        self.imS = cv2.resize(norm_image, (x,y))  
        self.d = abs(cv2.resize(disparity, (x,y))/multiple)
        self.cache = copy.deepcopy(self.imR)
        cv2.imshow('disparity',self.imS)
        while(1):
            cv2.namedWindow('img_L')
            cv2.namedWindow('img_R')
            cv2.imshow('img_L',self.imL)
            cv2.imshow('img_R',self.imR)
            cv2.setMouseCallback('img_L', self.draw_circle)
            if cv2.waitKey(20) & 0xFF == 27:
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if int(self.d[y][x])!=0:
                self.imR = copy.deepcopy(self.cache)
                cv2.imshow('img_R', self.imR)
                cv2.circle(self.imR,(x-int(self.d[y][x]), y), 4, (255, 255, 0), -1)

                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())

        
        
        
        
        
            


# In[ ]:




