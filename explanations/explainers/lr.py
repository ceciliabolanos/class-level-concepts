import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LRExplainer():

    def __init__(self, perturbations, scores, neighborhood, kernel_width=0.25):
        self.perturbations = perturbations
        self.scores = scores
        self.neighborhood = neighborhood
        self.kernel_width = kernel_width
        self.kernel_fn = partial(LRExplainer.kernel, kernel_width=float(self.kernel_width))
        self.model = None

    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    
    def get_feature_importances(self, 
                                weighting=None,
                                regularization=None,
                                alpha=1.0):
        y = np.array(self.scores)
        Xs = np.array(self.perturbations)
        features = range(Xs.shape[1])

        if weighting:
            distances = np.array(self.neighborhood)
            min_non_zero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 1e-8
            distances = np.maximum(distances, min_non_zero_dist * 0.1)
            weights = self.kernel_fn(distances)
            weights = np.maximum(weights, 1e-8)
            weights = weights / np.sum(weights)
        else:
            distances = self.neighborhood 
            weights = np.ones_like(distances)

        model_regressor = None
        if regularization == 'lasso':
            model_regressor = Lasso(alpha=alpha, fit_intercept=True)
        elif regularization == 'ridge':
            model_regressor = Ridge(alpha=alpha, fit_intercept=True)
        else:
            model_regressor = LinearRegression(fit_intercept=True)
        
        model_regressor.fit(Xs[:, features], y, sample_weight=weights)
        local_pred = model_regressor.predict(Xs[0, features].reshape(1, -1))
        self.model = model_regressor
        return model_regressor.coef_, local_pred
    
    def metrics(self, inputs, y_true):
        y_pred = self.model.predict(inputs)
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        return mse, r2, mae, y_pred
    