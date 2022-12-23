# For data
import pandas as pd
# To show final graph
import matplotlib.pyplot as plt
# To random w and b start values
import random
from json import load

# Reading our data (Students sutdy time and students scores)
#data = pd.read_csv('/home/adam/Datasets/student_scores.xls')
data = load(open('data.txt', 'r'))
plt.scatter(data[0], data[1], color='black')
plt.show()

# We round our w and b to break loop if our prev w, b = w, b
r = 3

# Our function to calculate the loss
def loss(data, w, b):
    loss = 0

    n = len(data)
    for i in range(n):
        x = data[0][i]
        y = data[1][i]
        # Loss formula
        loss += (y - (w * x + b)) ** 2

    return (loss / n) 

# Our gradiend descent function to calculate best w and b values
def gradient_descent(data, w_cur, b_cur, l):
    w_grad = 0
    b_grad = 0

    n = len(data)

    for i in range(n):
        x = data[0][i]
        y = data[1][i]
        # Our loss function derivative with respect to w
        w_grad += -(2/n) * x * (y - (w_cur * x + b_cur))
        # Our loss function derivative with respect to b
        b_grad += -(2/n) * (y - (w_cur * x + b_cur))

    # Calculationg new w and b values
    w = w_cur - w_grad * l
    b = b_cur - b_grad * l
    
    # Rounding it to r decimal places
    w = round(w, r)
    b = round(b, r)

    return w, b

# How much times we will do gradient descent to calculate best w and b
epochs = 100
# Random starting values
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)
# Learning rate
l = 0.0002

for i in range(epochs):
    # For amount of epochs
    e = i+1
    # Our new w and b = grad_desc(currren w and b)
    w, b = gradient_descent(data, w, b, l)
    print(f'epoch: {e}\r', end='')
    # If w and b (close to r decimal places) is equal to new w and b
    #if (round(w, r), round(b, r)) == gradient_descent(data, w, b, l):
        # We break the loop (learning)
        #break

    # Showing linear regression line every 5 epochs (to see how it changes)
    '''if i % 5 == 0:
        plt.scatter(data[0], data[1], color='black')
        plt.plot(list(range(1, 11)), [w * x + b for x in range(1, 11)], color='red')
        plt.title(f'Epoch: {i}')
        #plt.show()'''

# Printing out w and b value, loss value and amount of epochs
print(f'w: {w}, b: {b}, loss: {loss(data, w, b)}, epohcs: {e}')

# Showing final graph
plt.scatter(data[0], data[1], color='black')
plt.plot(list(range(-60, 60)), [w * x + b for x in range(-60, 60)], color='blue', linewidth=5)
plt.title('Final line')
plt.xlabel(f'Epochs needed to train model: {e}')
plt.show()