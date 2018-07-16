# -*- coding: utf-8 -*-
# @Time    : 2018/07/02 3:36
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : 22.py
# @Software: PyCharm

import numpy as np
import numpy.linalg as LA

def compute_norm():
    mat = np.matrix([[2,3]])
    inv_mat = np.linalg.inv(mat)
    print (inv_mat)
    print(LA.norm(mat, 1))  # 1范数


def vector_norm():
    a = np.arange(2) - 4
    print(a)
    print ( LA.norm(a,np.inf)) #无穷范数)
    print ( LA.norm(a,-np.inf))
    print ( LA.norm(a,1)) #1范数
    print ( LA.norm(a,2)) #2范数

def matrix_norm():
    a = np.arange(9) - 4
    b = a.reshape(3,3)
    b_t = np.transpose(b)
    b_new = np.dot(b_t,b) #b_new矩阵为b^t * b
    x = np.linalg.eigvals(b_new) #求b_new矩阵的特征值
    print ( x)
    print ( LA.norm(b,1)) #列范数
    print ( LA.norm(b,2) )#谱范数,为x里最大值开平方
    print ( LA.norm(b,np.inf) )#无穷范数，行范数
    print ( LA.norm(b,"fro") )#F范数
# compute_norm()
vector_norm()
#
# matrix_norm()