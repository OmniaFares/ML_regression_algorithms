import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('heart.csv')
x = np.c_[data['trestbps'], data['chol'], data['thalach'], data['oldpeak']]
y = data['target']


percent = (int)(y.size * 0.8)
x_train = x[:percent]
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
y_train = np.array(y[:percent])

x_test = x[percent:]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]
y_test = np.array(y[percent:])


theta = np.zeros(np.size(x, 1) + 1)
alpha = 0.1
iterations = 1000


def costfunction(theta,x,y):
    h = hypothesis(x,theta)
    predictions1 = -y * np.log(h)
    predictions2 = (1 - y) * np.log(1-h)
    errors = (predictions1 - predictions2)
    J = 1 / (y.size) * sum(errors)
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
    y_predict = hypothesis(x, theta)
    print(y_test, y_predict)
    MSE = (1 / y.size) * np.sum(((y_predict - y_test) ** 2))
    acc = np.sum(np.equal(y_test, y_predict)) / len(y_predict)
    return acc , MSE


theta, past_costs, past_thetas = gradient_descent(x_train, y_train, theta, iterations, alpha)
print(theta)

acc, MSE = predict(x_test, y_test, theta)
print("error = ", MSE)
print("acc = ", acc)

############### Testing #################
model = LogisticRegression(random_state=0, solver='lbfgs').fit(x, y)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y_test))


