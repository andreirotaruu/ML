import numpy as np
import pandas as pd

def generate_dataset():
    np.random.seed(42)

    X = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 2, 100)
    
    y = 2 * X + 5 + noise

    data = pd.DataFrame({
        "x": X,
        "y": y
    })

    data.to_csv("/Users/andre/Documents/ML/linear_regression/linear_data.csv", index=False)


if __name__ == '__main__':
    generate_dataset()