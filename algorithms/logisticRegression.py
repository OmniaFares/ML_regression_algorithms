import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('heart.csv')
x = data[['trestbps', 'chol', 'thalach', 'oldpeak']]
y = data['target']
x = (x - x.mean()) / (x.std())

TrainSize = int(len(x) * 0.8)
x_train = x[:TrainSize]
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
y_train = np.array(y[:TrainSize])

x_test = x[TrainSize:]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]
y_test = np.array(y[TrainSize:])


theta = np.zeros(np.size(x, 1) + 1)
alphaArr = [0.001, 0.00005, 0.0003]
iterations = 1000


def costfunction(theta,x,y):
    h = hypothesis(x,theta)
    predictions1 = y*np.log(h)
    predictions2 = (1-y) * np.log(1-h)
    errors = (predictions1 + predictions2)
    J = (-1 / (y.size)) * np.sum(errors)
    return J


def hypothesis(x, theta):
    ex = np.dot(x, theta)
    return 1 / (1 + np.exp(-ex))


def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = []
    for i in range(iterations):
        h = hypothesis(x, theta)
        j = costfunction(theta, x, y)
        past_costs.append(j)
        diff = np.dot(x.T, h - y)
        theta = theta - alpha * diff
        past_thetas.append(theta)

    return theta, past_costs, past_thetas


def predict(x, y_test, theta):
    h = hypothesis(x, theta)
    y_predict = np.where(h >= .5, 1, 0)
    print("Actual Data  : ", y_test)
    print("Predicted Data  : ", y_predict)
    acc = np.sum(np.equal(y_test, y_predict)) / len(y_predict)
    return acc

for alpha in alphaArr :
    print("For Alpha = " , alpha)
    print("------------------------------------------")
    theta, past_costs, past_thetas = gradient_descent(x_train, y_train, theta, iterations, alpha)
    print("Theta Values : ", theta)

    acc = predict(x_test, y_test, theta)
    print("Accuracy = ", acc)

    theta = np.zeros(np.size(x, 1) + 1)
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(past_costs)
    plt.show()
    print("------------------------------------------")



