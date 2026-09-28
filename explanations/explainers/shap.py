import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SHAPExplainer():
    def __init__(self, perturbations, scores, neighborhood=None):
        self.perturbations = np.array(perturbations)
        self.scores = np.array(scores)
        self.neighborhood = neighborhood
        self.coeffs = None  
        self.b0 = None  

    def shap_kernel_weight(self, m, z):
        """Calcula el peso del Kernel SHAP."""
        if z == 0 or z == m:
            return 1e10 
        
        log_comb = gammaln(m + 1) - gammaln(z + 1) - gammaln(m - z + 1)
        log_weight = np.log(m - 1) - log_comb - np.log(z) - np.log(m - z)
        return np.exp(log_weight)
    
    def pi_x_for_list(self, vectors):
        weights = []
        for x in vectors:
            m = len(x)
            z = np.sum(x)
            weight = self.shap_kernel_weight(m, z)
            weights.append(weight)

        if len(set(weights)) == 1:
            weights = [1] * len(weights)
        mean = sum(weights)/len(weights)
        weights = [w / mean for w in weights]
        return np.array(weights)


    def get_feature_importances(self, weighting=True, empty_constraint=None):
        y_original_pred = self.scores[0]
        y_perturbations = self.scores[1:]
        
        X_perturbations = self.perturbations[1:]
        
        if weighting:
            weights = self.pi_x_for_list(X_perturbations)
        else:
            weights = np.ones(len(y_perturbations))
            
        if empty_constraint is not None:
            X = X_perturbations
            y_target = y_perturbations - empty_constraint 
            W = np.diag(weights)
            target_sum = y_original_pred - empty_constraint

            XTWX = X.T @ W @ X
            ones = np.ones((X.shape[1], 1))

            top_block = np.hstack([XTWX, ones])
            bottom_block = np.hstack([ones.T, [[0]]])
            A = np.vstack([top_block, bottom_block])

            # Vector del lado derecho (b)
            XTWy = X.T @ W @ y_target
            b = np.concatenate([XTWy, [target_sum]])

            try:
                solution = np.linalg.solve(A, b)
                self.coeffs = solution[:-1] 
                
            except np.linalg.LinAlgError:
                print("Matriz singular, usando fallback...")

            self.b0 = empty_constraint

        else:
            ones_col = np.ones((X_perturbations.shape[0], 1))
            X_aug = np.hstack([X_perturbations, ones_col])
            
            W = np.diag(weights)
          
            XTWX = X_aug.T @ W @ X_aug
            constraint_vec = np.ones((X_aug.shape[1], 1))

            top_block = np.hstack([XTWX, constraint_vec])
            bottom_block = np.hstack([constraint_vec.T, [[0]]])
            
            A = np.vstack([top_block, bottom_block])

           
            XTWy = X_aug.T @ W @ y_perturbations
            b_vec = np.concatenate([XTWy, [y_original_pred]])

            try:
                solution = np.linalg.solve(A, b_vec)

                constrained_coeffs = solution[:-2] 
                b0 = solution[-2]

            except np.linalg.LinAlgError:
                print("Aviso: Matriz singular, usando mínimos cuadrados aproximados.")
                solution, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                constrained_coeffs = solution[:-2]
                b0 = solution[-2]

            self.coeffs = constrained_coeffs
            self.b0 = b0
            
        return self.coeffs, None

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.dot(X, self.coeffs) + self.b0
    
    def metrics(self, inputs, y_true):
        y_pred = self.predict(inputs)
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        return mse, r2, mae, y_pred
    
