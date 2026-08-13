from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#load data in df
df = pd.read_csv('linear_regression/linear_data.csv')

#convert to array and reshape to 2d array
X = np.array(df['x']).reshape(-1, 1)
y = np.array(df['y']).reshape(-1, 1)


def linear_regression_pipeline():
    pipe = make_pipeline(
        StandardScaler(), 
        LinearRegression()
    )

    #split training and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)

    x_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_values = pipe.predict(x_values)

    plt.scatter(X, y)
    plt.plot(x_values, y_values)
    plt.show()

def linear_regression():

    #without scaling
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model.fit(x_train, y_train)
    
    #print m and b compare against from scratch impl
    print(model.coef_)
    print(model.intercept_)


linear_regression()