import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('house_data.csv')
x = data[['grade', 'bathrooms', 'lat', 'sqft_living', 'view']]
y = data['price']

# initializing data
x = (x - x.mean()) / x.std()
TrainSize = int(len(x) * 0.8)  # No. of data points
x_train = x[:TrainSize]
x_train = np.c_[np.ones(x_train.shape[0]), x_train]  # final training set
y_train = np.array(y[:TrainSize])

x_test = x[TrainSize:]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]  # final testing set
y_test = np.array(y[TrainSize:])

"""TrainSize = int(len(x) * 0.8)  # No. of data points
x_train = x[:TrainSize]
x_train = (x_train - x_train.mean()) / x_train.std()  # Normalizing train data
x_train = np.c_[np.ones(x_train.shape[0]), x_train]  # final training set
y_train = np.array(y[:TrainSize])

x_test = x[TrainSize:]
x_test = (x_test - x_test.mean()) / x_test.std()  # Normalizing test data
x_test = np.c_[np.ones(x_test.shape[0]), x_test]  # final testing set
y_test = np.array(y[TrainSize:])"""

# initializing operators
theta = np.zeros(np.size(x, 1) + 1)
alpha = 0.007  # Step size
iterations = 2500  # No. of iterations


def hypothesis(xs, thetas):
    return np.dot(xs, thetas)


def cost_function(h, ys, m):
    div = h - ys
    j = np.dot(div.T, div) / (2 * m)
    return j


def gradient_descent(a, m, xs, ys, thetas, iters):
    past_costs = []
    past_thetas = []
    for i in range(iters):
        h = hypothesis(xs, thetas)
        diff = np.dot(xs.T, (h - ys))
        thetas = thetas - ((a / m) * diff)
        past_thetas.append(thetas)
        j = cost_function(h, ys, m)
        past_costs.append(j)


    return thetas, past_thetas, past_costs


def predict(xs, ys, thetas):
    y_predict = hypothesis(xs, thetas)
    div = np.subtract(ys, y_predict)
    MSE = np.dot(div.T, div) / len(xs)
    return MSE


(final_theta, thetas, costs) = gradient_descent(alpha, TrainSize, x_train, y_train, theta, iterations)

for i in range(len(final_theta)):
    print('theta', i, '= ', final_theta[i])

total_error = predict(x_test, y_test, final_theta)
print('error for test data = ', total_error)

plt.plot(costs)
plt.title('change of  J(theta)')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()

