# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# ################################################################################
#
# print('Load Image')
#
#
# imgFile = 'D:\\GitHub\\tensorflow_from_my_notebook\\4.computer version\\pic\\m_1342505545217.jpg'
#
# # load an original image
# img = cv2.imread(imgFile)
# ################################################################################
#
# # color value range
# cRange = 256
#
# # convert color space from bgr to gray
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# ################################################################################
#
# # pyramid level
# level = 5
#
# # original image at the bottom of gaussian pyramid
# higherResoGauss = img
# plt.subplot(2, 1 + level, 1), plt.imshow(higherResoGauss), plt.title('Gaussian Level ' + '%d' % level), plt.xticks(
#     []), plt.yticks([])
#
# for l in range(level):
#
#     rows, cols, channels = higherResoGauss.shape
#
#     # delete last odd row of gaussian image
#     if rows % 2 == 1:
#         higherResoGauss = higherResoGauss[:rows - 1, :]
#     # delete last odd column of gaussian image
#     if cols % 2 == 1:
#         higherResoGauss = higherResoGauss[:, :cols - 1]
#
#     # gaussian image
#     lowerResoGauss = cv2.pyrDown(higherResoGauss)
#     # even rows and cols in up-sampled image
#     temp = cv2.pyrUp(lowerResoGauss)
#     print(higherResoGauss.shape, temp.shape)
#
#
#     # laplacian image
#     lowerResoLap = higherResoGauss - temp
#
#     # display gaussian and laplacian pyramid
#     plt.subplot(2, 1 + level, l + 2), plt.imshow(lowerResoGauss), plt.title(
#         'Gaussian Level ' + '%d' % (level - l - 1)), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 1 + level, 1 + level + l + 2), plt.imshow(lowerResoLap), plt.title(
#         'Laplacian Level ' + '%d' % (level - l - 1)), plt.xticks([]), plt.yticks([])
#
#     higherResoGauss = lowerResoGauss
# ################################################################################
#
# # display original image and gray image
# plt.show()
# ################################################################################
#
# print('Goodbye!')


imgFile = 'D:\\GitHub\\tensorflow_from_my_notebook\\4.computer version\\pic\\m_1342505545217.jpg'
import cv2
import numpy as np

def gaussian_pyr(img,lev):
    img = img.astype(np.float)
    g_pyr = [img]
    cur_g = img;
    for index in range(lev):
        print(index)
        cur_g = cv2.pyrDown(cur_g)
        g_pyr.append(cur_g)
    return g_pyr


def laplacian_pyr(img,lev):
    img = img.astype(np.float)
    g_pyr = gaussian_pyr(img,lev)
    l_pyr = []
    for index in range(lev):
        cur_g = g_pyr[index]
        cur_w,cur_h = np.shape(cur_g)
        next_g = cv2.pyrUp(g_pyr[index+1],dstsize=(cur_h,cur_w))
        cur_l = cv2.subtract(cur_g,next_g)
        l_pyr.append(cur_l)
    l_pyr.append(g_pyr[-1])
    return l_pyr

def lpyr_recons(l_pyr):
    lev = len(l_pyr)
    cur_l = l_pyr[-1]
    for index in range(lev-2,-1,-1):
        #print(index)
        next_w,next_h = np.shape(l_pyr[index])
        cur_l = cv2.pyrUp(cur_l,dstsize=(next_h,next_w))
        next_l = l_pyr[index]
        cur_l = cur_l + next_l
    return cur_l
import imageio
import matplotlib.pyplot as plt
img = imageio.imread(imgFile)
# img = luminance(img)

m = gaussian_pyr(img,5)
for i in range(len(m)):
    plt.imshow(m[i],cmap='gray')
    plt.show()



g = laplacian_pyr(img,5)
for i in range(len(g)):
    plt.imshow(g[i],cmap='gray')
    plt.show()

t = lpyr_recons(g)
plt.imshow(t,cmap='gray')
plt.show()

