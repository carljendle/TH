import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.linear_model import LinearRegression



data = pd.read_csv("linear_data.csv")

x_vals = np.asarray(data["X"])
y_vals = np.asarray(data["Y"])


def linear_regression(x_vals: np.array, y_vals: np.array, learning_rate: float = 0.002, intercept: float = 0, slope: float = 0, max_iter: int = 100):
    """
    Performs linear regression on input data with gradient descent. Returns expression for the line.
    """
    n_elements = len(x_vals)

    threshold = 10
    for i in range(max_iter):
        #Gör om predictions för y i varje iteration
        y_pred = slope*x_vals + intercept
        #Ta fram uttryck för slopes derivata baserat på predictions och ursprungsdatan
        dSlope = -(2/n_elements)*np.dot(x_vals,(y_vals - y_pred))
        #Ta fram uttryck för intercepts derivata baserat på predictions och ursprungsdatan
        dIntercept = -(2/n_elements)*np.sum(y_vals - y_pred)

        if np.abs(dIntercept) > threshold:
            dIntercept = 1
        if np.abs(dSlope) > threshold:
            dSlope = 1




        #Uppdatera slope och intercept
        slope -= learning_rate*dSlope
        intercept -=  learning_rate*dIntercept


    return slope, intercept

def analytical_regr(x_vals: np.array, y_vals: np.array):
    '''
    Analytical solution for one dimensional Linear Regression.
    '''
    n_elements = len(x_vals)
    x_vals_squared = sum([x**2 for x in x_vals])
    y_vals_squared = sum([y**2 for y in y_vals])
    x_sum = sum(x_vals)
    y_sum = sum(y_vals)
    xy_vals = sum([x*y for x,y in zip(x_vals, y_vals)])


    slope = (n_elements*xy_vals-x_sum*y_sum)/(n_elements*x_vals_squared - x_sum**2)
    intercept = (y_sum - slope*x_sum)/n_elements

    return slope, intercept




print(f"Slope, intercept gradient descent: {linear_regression(x_vals, y_vals, max_iter = 3000, learning_rate = 0.6)}")
print(f"Slope, intercept analytical: {analytical_regr(x_vals, y_vals)}")




print(f"Original shape for x_vals: {x_vals.shape}")
print(f"Original shape for y_vals: {y_vals.shape}")
#Reshapa för att få in ordentligt i sklearn
x_vals = x_vals.reshape(-1,1)
y_vals = y_vals.reshape(-1,1)

print(f"New shape for x_vals: {x_vals.shape}")
print(f"New shape for y_vals: {y_vals.shape}")

reg = LinearRegression().fit(x_vals, y_vals)

print(f"Slope, intercept sklearn:{reg.coef_}")
print(f"Intercept sklearn: {reg.intercept_}")
