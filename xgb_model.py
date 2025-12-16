import xgboost as xgb
import numpy as np

class xgb_model:
    def __init__(self, p, hyperparams):
        self.md = xgb.XGBRegressor(
            **hyperparams
        )
        self.p = p
    
    def train_model(self, train):
        X_train, y_train = self._create_lag_features(data=train, n_lags=self.p)
        self.md.fit(X_train, y_train)
    
    def _create_lag_features(self, data, n_lags):
        """Create lag features for time series data.
        
        Args:
            data: Time series data
            n_lags: Number of lags to create
            
        Returns:
            Tuple of (X, y) where X contains lag features and y contains target values
        """
        X = np.array([
            data[i - n_lags:i].values for i in range(n_lags, len(data))
        ])
        y = data[n_lags:]
        return X, y

    def forecast(self, train, n_forecast):
        """Generate rolling predictions using only training data and previous predictions.
        
        Args:
            train: Training data
            n_forecast: Number of steps to forecast
            
        Returns:
            Array of predicted values
        """
        preds = np.zeros(n_forecast)
        last_values = train[-self.p:].values.copy()

        for i in range(n_forecast):
            X_pred = last_values.reshape(1, -1)
            
            y_pred = self.md.predict(X_pred)[0]
            preds[i] = y_pred
            
            last_values = np.append(last_values[1:], y_pred)
        
        return preds

    def predict(self, full_series, train_size):
        """Generate predictions using true values (up-to-date approach).
        
        Args:
            full_series: Complete time series data
            train_size: Size of training set (start index for predictions)
            
        Returns:
            Array of predicted values
        """
        X_test, _ = self._create_lag_features(full_series[train_size - self.p:], self.p)
        return self.md.predict(X_test)
