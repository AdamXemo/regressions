import numpy as np
from matplotlib import pyplot as plt
from json import dump

d = 0.5
a = -0.6
b = 7
c = -25.3

xs = np.array([float(i/10) for i in range(-100,101)])
ys = np.array(list(map((lambda x: d*x**3 + a*x**2 + b*x + c), xs)))
plt.scatter(xs, ys)
xs += np.random.normal(0.0, scale=5.0, size=xs.shape)
ys = np.array(list(map((lambda x: d*x**3 + a*x**2 + b*x + c), xs)))
ys += np.random.normal(0.0, scale=180.0, size=ys.shape)
plt.scatter(xs, ys)

print(ys)

dump([list(xs), list(ys)], f:=open("data.txt", "w"))
f.close()

plt.show()