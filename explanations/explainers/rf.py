from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RFExplainer:
    def __init__(self, perturbations, scores, neighborhood, 
                 n_estimators=100, max_depth=None, max_features=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.perturbations = perturbations
        self.scores = scores
        self.neighborhood = neighborhood
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
    def get_feature_importances(self, method='gini'):
        y = self.scores
        distances = self.neighborhood

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=16
        ) 
        self.model.fit(self.perturbations, y, sample_weight=distances)

        local_pred = 0
        if method == 'gini':
            importances = self.model.feature_importances_

        return importances, local_pred
    
    def metrics(self, inputs, y_true):
        y_pred = self.model.predict(inputs)
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, r2, mae, y_pred

    