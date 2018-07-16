# -*- coding: utf-8 -*-
# @Time    : 2018/01/11 1:33
# @Author  : Yu Bowen
# @Site    : www.ybwsfl.xin
# @File    : PIL.Image转换成OpenCV格式.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), NJUST
# @Contact : yubowen_njust@163.com
# @Modified : 
import cv2
from PIL import Image
import numpy

image = Image.open("plane.jpg")
image.show()
img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
cv2.imshow("OpenCV", img)
cv2.waitKey()