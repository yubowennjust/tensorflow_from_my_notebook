import os
from matplotlib import pyplot as plot
from skimage import io,transform
img = io.imread('D:/GitHub/tensorflow_from_my_notebook/4.computer version/images/yubowen.jpg')
print(img.shape)

io.imshow(img)
plot.show()

print(type(img))  #显示类型
print(img.shape)  #显示尺寸
print(img.shape[0])  #图片高度
print(img.shape[1])  #图片宽度
print(img.shape[2])  #图片通道数
print(img.size)   #显示总像素个数
print(img.max())  #最大像素值
print(img.min())  #最小像素值
print(img.mean()) #像素平均值
print(img[0][0])#图像的像素值