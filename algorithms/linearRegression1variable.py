import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('house_data.csv')
x = data['sqft_living']
y = data['price']

"""percent = (int) (x.size * 0.8)
x_train = x[:percent]
RangeX_train = x_train.max() - x_train.min()
newXtrain = []
for i in range (len((newXtrain))):
    newXtrain.insert(i, (x_train[i] - x_train.mean()) / x_train.std())
x_train = np.array(x_train)
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
y_train = np.array(y[:percent])


x_test = x[percent:]
RangeX_test = x_test.max() - x_test.min()
newXtest = []
for i in range (len((newXtest))):
    newXtest.insert(i, (x_test[i] - x_test.mean()) / x_test.std())
x_test = np.array(x_test)
x_test = np.c_[np.ones(x_test.shape[0]), x_test]
y_test = np.array(y[percent:])
print(x_train)
print(y_train)"""

RangeX = x.max() - x.min()
newX = []
for i in range(len(x)):
    newX.insert(i, (x[i] - x.mean()) / x.std())

x = np.array(newX)
percent = (int) (x.size * 0.8)
x_train = x[:percent]
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
y_train = y[:percent]

x_test = x[percent:]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]
y_test = y[percent:]

alpha = 0.05  # Step size
iterations = 2500  # No. of iterations
m = y.size  # No. of data points
theta = [0, 0]


# GRADIENT DESCENT
def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = []
    for i in range(iterations):
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        diff = (alpha * (1 / m)) * np.dot(x.T, error)
        j = (1 / (2 * m)) * (error ** 2)
        past_costs.append(j)
        theta = theta - diff
        past_thetas.append(theta)

    return theta, past_thetas, past_costs


theta, past_thetas, past_costs = gradient_descent(x_train, y_train, theta, iterations, alpha)
print("Gradient Descent: theta0 = {:.2f}, theta1 = {:.2f}".format(theta[0], theta[1]))

def predict(x,y,theta):
    y_predcit = np.dot(x,theta)
    MSE = (1/y.size) * np.sum(((y_predcit - y)**2))

    return MSE

total_error = predict(x_test, y_test, theta)
print("error = ", total_error)

#show cost function
"""# Pass the relevant variables to the function and get the new values back...
# Print the results...
print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))

# Plot the cost function...
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()"""


plt.title('Living VS price')
plt.xlabel('Living Area sqft (normalised)')
plt.ylabel('Sale Price ($)')
plt.scatter(x_train[:, 1], y_train)
best_y = [theta[0] + theta[1]* xx for xx in x_train]
plt.plot(x_train, best_y, '-', color='red')
plt.show()
