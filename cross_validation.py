from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import config
import utils
import time

# FIX P = 20
p = 20

# PJME:
PJME = utils.load_dataset(config.DATASETS[0]['name'], config.DATASETS[0]['parser'])
PJME_train = PJME[:int(0.8 * len(PJME))].loc[:, 'PJME_MW']
X_train, y_train = PJME_train.iloc[:-p], PJME_train.iloc[p:]

param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'random_state': [42],
    'verbosity': [0]
}

t0 = time.time()
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
print(f'time taken: {time.time() - t0}')
print("Results for PJME:")
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}

# Sunspots:

sunspots = utils.load_dataset(config.DATASETS[1]['name'], config.DATASETS[1]['parser'])
sunspots_train = sunspots[:int(0.8 * len(sunspots))].loc[:, 'sunspot_number']
X_train, y_train = sunspots_train.iloc[:-p], sunspots_train.iloc[p:]

param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'random_state': [42],
    'verbosity': [0]
}

t0 = time.time()

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
print(f'time taken: {time.time() - t0}')

print("Results for Sunspots:")
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.6}
