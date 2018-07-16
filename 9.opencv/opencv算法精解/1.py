# -*- coding: utf-8 -*-
# @Time    : 2018\\07\\05 23:52
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : 1.py
# @Software: PyCharm


#导入cv模块
import cv2 as cv
#读取图像，支持 bmp、jpg、png、tiff 等常用格式
img = cv.imread("D:/Pictures/m_1342505545217.jpg")
#创建窗口，并显示图像
cv.namedWindow("image")
cv.imshow("image",img)

b=img[:,:,0]
g=img[:,:,1]
r=img[:,:,2]

cv.imshow("b",b)
cv.imshow("g",g)
cv.imshow("r",r)

cv.waitKey(0)
#释放窗口
cv.destroyAllWindows()
