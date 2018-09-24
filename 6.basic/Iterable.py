from collections import Iterable

print(isinstance([],Iterable))

for x in [1, 2, 3, 4, 5]:
    pass

it = iter([1, 2, 3, 4, 5])
# 循环:
while True:
    try:
        # 获得下一个值:
        x = next(it)
    except StopIteration:
        # 遇到StopIteration就退出循环
        break