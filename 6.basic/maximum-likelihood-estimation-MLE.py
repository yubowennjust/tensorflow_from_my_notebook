# -*- coding: utf-8 -*-
# @Time    : 2018/06/25 20:13
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : maximum-likelihood-estimation-MLE.py
# @Software: PyCharm

import numpy as np
import random
import matplotlib.pyplot as plt


#生成样本
x=np.arange(-50,50,0.2)
print(x)

array_x=[]
array_y=[]

for a in x:
    linex=[1]
    linex.append(a)
    array_x.append(linex)
    array_y.append(0.5*a+3.5+random.uniform(0,1)*4*np.sin(2*a))

print(array_x)
print(array_y)

xmat=np.mat(array_x)
ymat=np.mat(array_y).T
xtx=xmat.T*xmat
w=xtx.I*xmat.T*ymat

y=xmat*w

plt.title("linear regression")
plt.xlabel("independent variable")
plt.ylabel("dependent variable")
plt.plot(x,array_y,color='g',linestyle='',marker='.')
plt.plot(x,y,color='r',linestyle='-',marker='')
plt.show()