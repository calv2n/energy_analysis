from statsmodels.tsa.ar_model import AutoReg
import numpy as np

class ar_model:
    def __init__(self, p):
        self.p = p
        self.trained = False
    
    def train_model(self, train):
        self.md = AutoReg(endog=train, lags=self.p).fit()
        self.trained = True
    
    def forecast(self, train, n_forecast):
        """Generate rolling predictions using only training data and previous predictions.
        
        Args:
            train: Training data
            n_forecast: Number of steps to forecast
            
        Returns:
            Array of predicted values
        """
        if not self.trained:
            raise Exception("Must train model to get forecast")
        train_size = len(train)
        preds = self.md.get_prediction(
            start=train_size, 
            end=train_size + n_forecast - 1
        ).predicted_mean
        return preds
    
    def predict(self, full_series, train_size):
        """Generate predictions using true values (up-to-date approach).
        
        Args:
            full_series: Complete time series data
            train_size: Size of training set (start index for predictions)
            
        Returns:
            Array of predicted values
        """
        if not self.trained:
            raise Exception("Must train model first")

        test_size = len(full_series) - train_size
        data_block = full_series[train_size - self.p:]
        X_test = np.array([
            data_block[i : i + self.p][::-1] 
            for i in range(test_size)
        ])

        params = self.md.params
        intercept = params[0]
        coeffs = params[1:]
        preds = intercept + np.dot(X_test, coeffs)
        return preds
