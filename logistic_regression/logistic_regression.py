import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def main():
    print("Logistic Regression")

#sigmoid function 
def sigmoid(z):
    return 1 / 1 + np.exp(-z)

#function for calculating the gradient
def calculate_gradient(X, y, theta):
    m = y.size #number of instances
    return (X.T @ (sigmoid(X @ theta) - y) / m)

#implementing gradient descent 
def gradient_descent(X, y, alpha=.1, num_iter=100, tol=1e-7):
    #initialize col of ones for bias in features
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    #initialize [bias, weight1, weight2, ...]
    theta = np.zeros(X_b.shape[1])

    for i in range(num_iter):
        #calculate gradient(direction of cross entropy loss function increase)
        grad = calculate_gradient(X_b, y, theta)
        #move opposite to that direction
        theta -= alpha * grad

        #stop when we are close to the minimum
        if np.linalg.norm(grad) < tol:
            break
    return theta

#predicting the probability through sigmoid function
def predict_probablity(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

#getting our prediction 
def predict(X, theta, threshold = 0.5):
    return (predict_probablity(X, theta) > threshold).astype(int)

if __name__ == '__main__':
    main()