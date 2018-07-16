# -*- coding: utf-8 -*-
# @Time    : 2018/06/28 13:54
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : 1.py
# @Software: PyCharm


import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

x = np.linspace(-10, 10)
y = sigmoid(x)
tanh = 2 * sigmoid(2 * x) - 1

plt.xlim(-11, 11)
plt.ylim(-1.1, 1.1)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([-1, -0.5, 0.5, 1])

plt.plot(x, y, label="Sigmoid", color="blue")
plt.plot(2 * x, tanh, label="Tanh", color="red")
plt.legend()
plt.show()


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    y=(1.0-np.exp(-2*x))/(1.0+np.exp(-2*x))
    return y

def relu(x):
    y=x.copy()
    y[y<0]=0
    return y

def elu(x,a):
    y=x.copy()
    for i in range(y.shape[0]):
        if y[i]<0:
            y[i]=a*(np.exp(y[i]-1))
    return y

x=np.linspace(-3,3)
y_sigmoid=sigmoid(x)
y_tanh=tanh(x)
y_relu=relu(x)
y_elu=elu(x,0.25)
# plt.plot(x, y_sigmoid, 'r',label="y_sigmoid")
plt.plot(x, y_tanh, 'g',label="y_tanh")
# plt.plot(x, y_relu, 'b',label="y_relu")
# plt.plot(x, y_elu, 'k',label="Sigmoid")

plt.ylim([-3,3])
plt.legend()
plt.show()
