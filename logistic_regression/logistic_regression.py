import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def main():
    print("Logistic Regression")

def sigmoid(z):
    return 1 / 1 + np.exp(-z)

def calculate_gradient(X, y, theta):
    m = y.size #number of instances
    return (X.T @ (sigmoid(X @ theta) - y) / m)

def gradient_descent(X, y, alpha=.1, num_iter=100, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    for i in range(num_iter):
        grad = calculate_gradient(X, y, theta)
        theta -= alpha * grad

        if np.linalg.norm(grad) < tol:
            break
    return theta

def predict_probablity(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold = 0.5):
    return (predict_probablity(X, theta) > threshold).astype(int)

if __name__ == '__main__':
    main()