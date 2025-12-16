P_CANDIDATES = range(1, 41)
TEST_PROP = 0.1
DATASETS = [
    {
        'name': 'PJME_hourly',
        'target': 'PJME_MW',
        'parser': None 
    },
    {
        'name': 'SN_m_tot_V2.0',
        'target': 'sunspot_number',
        'parser': {
            'sep': ';',
            'header': None,
            'names': ['year', 'month', 'date_fraction', 'sunspot_number', 'std', 'num_obs', 'flag']
        }
    }
]
HYPERPARAMETERS = [ # Gathered from cross_validation.py
    {
        'n_estimators' : 100, 
        'learning_rate': 0.1, 
        'max_depth': 6, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'random_state': 42, 
        'verbosity': 0
    }, # PJME
    {
        'n_estimators' : 100, 
        'learning_rate': 0.1, 
        'max_depth': 6, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'random_state': 42, 
        'verbosity': 0
    } # SUNSPOTS 
]
