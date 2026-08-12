import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('linear_regression/linear_data.csv')

    m = 0
    b = 0
    L = .001
    epochs = 50000

    for i in range(epochs):
        if i % 500 == 0:
            print(i, loss_function(m, b, data))
    
        m, b = gradient_descent(m, b, data, L)


    print(f'm: {m}\nb: {b}')
    plt.scatter(data.x, data.y, color="black")
    x_values = np.linspace(data.x.min(), data.x.max(), 100)
    plt.plot(x_values, m * x_values + b, color="red")
    plt.show()


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y
        m_gradient += -2/n * (y - (m_now * x + b_now)) * x
        b_gradient += -2/n * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b
         
if __name__ == "__main__":
    main()