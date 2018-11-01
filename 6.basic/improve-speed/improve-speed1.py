import glob
import concurrent.futures
import os
import cv2
import time
import datetime


# count = 0
# start_time = time.time() # 记录时间
# for filename in os.listdir(r"D:\\Pictures\\test"):              #listdir的参数是文件夹的路径
#     print("D:\\Pictures\\test\\"+filename)
#     count += 1
#     print(count)
#     img = cv2.imread("D:\\Pictures\\test\\"+filename)
#     img = cv2.resize(img,(600,600))
# duration = time.time() - start_time #
# print(duration)

def load_and_resize(image_filename):
    img = cv2.imread(image_filename)

    img = cv2.resize(img, (600, 600))


image_files = glob.glob('D:\\Pictures\\test\\*.*')
print(len(image_files))
print(image_files[1])

for i in range(len(image_files)):
    print(image_files[i])
    img = cv2.imread(image_files[i])
    img = cv2.resize(img, (600, 600))





