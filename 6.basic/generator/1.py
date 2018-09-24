L = [x * x for x in range(10)]

g = (x * x for x in range(10))

print(L)
print(g)

for i in range(10):
    if i < 10 :
        print(next(g))