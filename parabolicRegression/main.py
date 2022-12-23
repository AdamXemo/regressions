import pandas as pd
from json import load
import matplotlib.pyplot as plt
import random

#data = load(open('/home/adam/Datasets/parabolicData.txt', 'r'))
data = load(open('data.txt', 'r'))

x = data[0]
y = data[1]
n = len(x)

def loss(a, b, c, x, y):
    loss = 0

    for i in range(n):
        # Loss formula
        loss += (y[i] - (a * x[i]**2 + b * x[i] + c)) ** 2

    return (loss / n)


def gradient_descent(a_current, b_current, c_current, x, y, learning_rate):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0

    for i in range(n):
        diff = a_current * x[i]**2 + b_current * x[i] + c_current - y[i]
        a_gradient += (2/n) * x[i]**2 * diff
        b_gradient += (2/n) * x[i] * diff
        c_gradient += (2/n) * diff
        
    a = a_current - a_gradient * learning_rate
    b = b_current - b_gradient * learning_rate  
    c = c_current - c_gradient * learning_rate

    return a, b, c

epochs = 6000
a = 0
b = 0
c = 0
learning_rate = 0.0000004

losses = []
for i in range(epochs):
    e = i+1
    print(f'epoch: {e}\r', end='')
    a, b, c = gradient_descent(a, b, c, x, y, learning_rate)
    if i % 100 == 0: 
        losses.append(loss(a, b, c, x, y))

print(f'a: {a}, b: {b}, c: {c}, loss: {loss(a, b, c, x, y)}')


plt.scatter(x, y, color='black')
plt.show()

plt.plot(list(range(-55, 56)), [a * x**2 + b * x + c for x in range(-55, 56)], color='red', linewidth=5)
plt.scatter(x, y, color='black')
plt.title('Final result')
plt.xlabel(f'Epochs needed to train model: {e}')
plt.show()

plt.plot(losses[1:])
plt.show()