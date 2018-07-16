import numpy as np

a = [1, 2, 3, 4]  #
b = np.array(a)  # array([1, 2, 3, 4])
type(b)  # <type 'numpy.ndarray'>

b.shape  # (4,)
print(b.argmax())  # 3
b.max()  # 4
b.mean()  # 2.5

c = [[1, 2], [3, 4]]  # 二维列表
d = np.array(c)  # 二维numpy数组
d.shape  # (2, 2)
d.size  # 4
print(d.max(axis=0))  # 找维度0，也就是最后一个维度上的最大值，array([3, 4])
print(d.max(axis=1))  # 找维度1，也就是倒数第二个维度上的最大值，array([2, 4])
d.mean(axis=0)  # 找维度0，也就是第一个维度上的均值，array([ 2.,  3.])
d.flatten()  # 展开一个numpy数组为1维数组，array([1, 2, 3, 4])
np.ravel(c)  # 展开一个可以解析的结构为1维数组，array([1, 2, 3, 4])

# 3x3的浮点型2维数组，并且初始化所有元素值为1
e = np.ones((3, 3), dtype=np.float)

# 创建一个一维数组，元素值是把3重复4次，array([3, 3, 3, 3])
f = np.repeat(3, 4)

# 2x2x3的无符号8位整型3维数组，并且初始化所有元素值为0
g = np.zeros((2, 2, 3), dtype=np.uint8)
g.shape  # (2, 2, 3)
h = g.astype(np.float)  # 用另一种类型表示

l = np.arange(10)  # 类似range，array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
m = np.linspace(0, 6, 5)  # 等差数列，0到6之间5个取值，array([ 0., 1.5, 3., 4.5, 6.])

p = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]]
)

np.save('p.npy', p)  # 保存到文件
q = np.load('p.npy')  # 从文件读取


a=np.array([1,2,3,4])
print(a)

import time

a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic = time.time()
c=np.dot(a,b)

toc=time.time()
print(c)
print("vectorized version:"+str(1000*(toc-tic)))

u=np.zeros((10,2))
print(u)

A = np.array([
    [56.0,0.0,4.4,68.0],
    [1.2,104.0,52.0,8.0],
    [1.8,135.0,99.0,0.9]
])

print(A)

cal = A.sum(axis=0)
print(cal)

percentage = 100*A/cal.reshape(1,4)
print(percentage)

a= np.random.rand(5)
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)
print(np.dot(a,a.T))

a= np.random.rand(5,1)
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)
print(np.dot(a,a.T))



c = np.array([
    [2],
    [3]
])
print(c.shape)
print(c*c)

