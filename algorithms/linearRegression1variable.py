import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('house_data.csv')

x = dataset['sqft_living']
y = dataset['price']
minValue = x.min()
maxValue = x.max()
#meanValue = trainDataX.iloc[:,1].mean()

x = (x - minValue)/(maxValue - minValue)

dataSize = int(0.8 * len(x))
trainDataX = x[:dataSize]
trainDataX = np.c_[np.ones(trainDataX.shape[0]), trainDataX]
testDataX = x[dataSize:]
testDataX = np.c_[np.ones(testDataX.shape[0]), testDataX]
trainDataY = y[:dataSize]
testDataY = y[dataSize:]

arrX = np.array(trainDataX)
arrY = np.array(trainDataY).flatten()
theta = np.array([0,0])

m = len(arrY)
iterations = 10000
alpha = [0.35, 0.1, 0.08]


def cost_function(x,y,theta,m):
    j = np.sum((x.dot(theta) - y) **2)/(2*m)
    return j


def gradient_descent(x,y,theta,alpha,iterations,m):
    history = [0] * iterations
    for iter in range(iterations):
        hyp = x.dot(theta)
        loss = hyp - y
        gradient = x.T.dot(loss)/m
        theta = theta - (alpha * gradient)
        cost = cost_function(x,y,theta,m)
        #print(cost)
        history[iter] = cost
    return theta,history


def predict(xs, ys, thetas):
    y_predict = np.dot(xs, thetas)
    div = np.subtract(ys, y_predict)
    MSE = np.dot(div.T, div) / len(xs)
    return MSE

for a in range(len(alpha)):
    print('###############################################################################################')
    print('with alpha =', alpha[a])
    (t, costHis) = gradient_descent(arrX, arrY, theta, alpha[a], iterations, m)

    for i in range(len(t)):
        print('\ttheta', i, '= ', t[i])

    total_error = predict(testDataX, testDataY, t)
    print('\n\terror for test data = ', total_error)

    plt.title(label= ('Cost Function J with alpha = ', alpha[a]))
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(costHis)
    plt.show()

    fitX = np.linspace(0, 1, 2)
    fitY = [t[0] + t[1] * xx for xx in fitX]
    plt.title(('Living VS price (alpha = ', alpha[a], ')'))
    plt.xlabel('Living Area sqft (normalised)')
    plt.ylabel('Sale Price ($)')
    plt.scatter(testDataX[:, 1], testDataY)
    plt.plot(fitX, fitY, color='red')
    plt.show()







