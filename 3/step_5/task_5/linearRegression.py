import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef = None


    def intercept(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate([ones, X], axis=1)


    def fit(self, X, y):
        X_b = self.intercept(X)
        #подсмотрел np.linalg.inv, сам первый раз взаимодействую с нужной делать обраткую матрицу (не учитывая pandas)
        self.coef = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        if self.coef is None:
            raise ValueError('сначало fit')
        
        X_b = self.intercept(X)
        return X_b @ self.coef