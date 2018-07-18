# -*- coding: utf-8 -*-
# @Time    : 2018/01/11 1:33
# @Author  : Yu Bowen
# @Site    : www.ybwsfl.xin
# @File    : OpenCV转换成PIL.Image格式.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), NJUST
# @Contact : yubowen_njust@163.com
# @Modified : 
import cv2
from PIL import Image
import numpy

img = cv2.imread("C:/Users/yubow/Documents/GitHub/tensorflow_from_my_notebook/10.network/2.AlexNET/images/llama.jpeg")
# cv2.imshow("OpenCV", img)
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
image.show()
cv2.waitKey()