"""
Explanation Generator - Main class for generating explanations

This module provides a comprehensive class-based system for generating 
explanations using various methods including surrogate models and CNN-based approaches.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch.nn as nn
from .lr import LRExplainer
from .rf import RFExplainer
from .shap import SHAPExplainer

from .explanation_config import (
    ExplanationConfig,
    TrainingMetrics,
    get_explanation_config,
    SurrogateTrainingConfig
)
import copy
from sklearn.preprocessing import StandardScaler
from explanations.auc_evaluator import AUCEvaluator

class ExplanationGenerator:
    """
    Main class for generating explanations from perturbation data.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ExplanationGenerator.
        
        Args:
            device: Device to use for computation ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"ExplanationGenerator initialized with device: {self.device}")
    
    def load_perturbation_data(
        self,
        path: str,
        num_batch: int,
        num_samples: int,
        id_to_explain: int,
        include_neighborhood: bool = True,
        test_batch_indices: List[int] = None
    ) -> Tuple[List, List, List, float]:
        """
        Load perturbation data from batch files.
        
        Args:
            path: Directory containing perturbation batches
            num_batch: Batch size
            num_samples: Total number of samples
            id_to_explain: Target class/output index
            include_neighborhood: Whether to include neighborhood similarity scores
            test_batch_indices: List of batch indices to exclude (test set)
            
        Returns:
            Tuple of (perturbations, scores, neighborhoods, score_real)
        """
        scores_combined = []
        neighborhood_combined = [] if include_neighborhood else None
        perturbations_combined = []
        score_real = None
        batch_idx = 0
        total = 0
        
        while total <= num_samples:
            if batch_idx in test_batch_indices:
                print(f"WARNING: BATCH {batch_idx} IS IN TEST SET, SKIPPING...")
                batch_idx += 1
                continue
            batch_path = f'{path}/batch{batch_idx}.npz'
            batch_idx += 1
            total += num_batch
            
            if not os.path.exists(batch_path):
                continue
                
            data = np.load(batch_path)
            
            # Add original sample on first batch
            if batch_idx == 1:
                perturbations_combined.append(np.ones_like(data['perturbations'][0]))
                scores_combined.append(data['score_real'][id_to_explain])
                if include_neighborhood:
                    neighborhood_combined.append(0)
            
            # Extract relevant scores for the target class
            scores_perturbed = [sublista[id_to_explain] for sublista in data['scores']]
            scores_combined.extend(scores_perturbed)
            perturbations_combined.extend(data['perturbations'].tolist())
            
            if include_neighborhood and 'neighborhood' in data:
                neighborhood_combined.extend(data['neighborhood'].tolist())
            
            if score_real is None:
                score_real = data['score_real'][id_to_explain]
        
        return perturbations_combined, scores_combined, neighborhood_combined, score_real
 
    def load_test_data(
        self,
        path: str,
        id_to_explain: int,
        test_batch_indices: List[int] = None
    ) -> Tuple[List, List]:
        """
        Load held-out test data from specific batch files.
        
        Args:
            path: Directory containing perturbation batches
            id_to_explain: Target class/output index
            test_batch_indices: List of batch indices to use for testing
            
        Returns:
            Tuple of (test_perturbations, test_scores)
        """
        if test_batch_indices is None:
            test_batch_indices = [134, 135, 136, 137, 138]
        
        perturbations_test = []
        scores_test = []
        
        for i in test_batch_indices:
            batch_path = f'{path}/batch{i}.npz'
            
            if not os.path.exists(batch_path):
                continue
            
            data = np.load(batch_path)
            
            # Add original sample on first test batch
            if i == test_batch_indices[0]:
                perturbations_test.append(np.ones_like(data['perturbations'][0]))
                scores_test.append(data['score_real'][id_to_explain])
            
            scores_perturbed = [sublista[id_to_explain] for sublista in data['scores']]
            scores_test.extend(scores_perturbed)
            perturbations_test.extend(data['perturbations'].tolist())
        
        return perturbations_test, scores_test
    
    def generate_surrogate_explanations(
        self,
        filename: str,
        id_to_explain: int,
        path: str,
        true_markers: List[List[float]],
        num_samples: int,
        num_batch: int,
        output_path: str, 
        explanation_config: Optional[ExplanationConfig] = None,
        perturbation_config: Optional[Dict[str, Any]] = None,
        test_batch_indices: Optional[List[int]] = None,
        audio: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = 16000
    ) -> Dict[str, Any]:
        """
        Generate explanations using surrogate model methods (SHAP, LR, RF, Tree).
        
        Args:
            filename: Audio filename being explained
            id_to_explain: Target class/output index
            path: Path to perturbation data
            true_markers: Ground truth markers/segments
            num_samples: Total number of samples for training
            num_batch: Batch size for data loading
            explanation_config: ExplanationConfig object defining which methods to use
            perturbation_config: Dictionary with perturbation configuration
            test_batch_indices: Batch indices to use for testing
            
        Returns:
            Dictionary with explanation results and metadata
        """
        # Use default config if none provided
        if explanation_config is None:
            explanation_config = get_explanation_config('surrogate_only')
        
        print(f"\n{'='*80}")
        print(f"Generating Surrogate Model Explanations")
        print(f"Config: {explanation_config.name}")
        print(f"{'='*80}\n")
        
        # Load training data
        print("Loading training data...")
        perturbations_combined, scores_combined, neighborhood_combined, score_real = \
            self.load_perturbation_data(
                path, num_batch, num_samples, id_to_explain, True, test_batch_indices
            )
        
        # Load test data
        print("Loading test data...")
        perturbations_test, scores_test = self.load_test_data(
            path, id_to_explain, test_batch_indices
        )

        train_tuples = {tuple(row) for row in perturbations_combined}
        indices_test_limpios = [
            i for i, row in enumerate(perturbations_test) 
            if tuple(row) not in train_tuples
        ]

        print(f"Test original: {len(perturbations_test)} muestras")

        perturbations_test = [perturbations_test[i] for i in indices_test_limpios]
        scores_test = [scores_test[i] for i in indices_test_limpios]
        
        print(f"Training samples: {len(scores_combined)}")
        print(f"Test samples: {len(scores_test)}\n")
        
        output_data = {
            "metadata": {
                "filename": filename,
                "id_explained": float(id_to_explain),
                "true_markers": true_markers,
                "true_score": float(score_real) if score_real is not None else None,
                "num_train_samples": len(scores_combined),
                "num_test_samples": len(scores_test)
            },
            "perturbation_config": perturbation_config or {
                "num_samples": num_samples,
                "num_batch": num_batch
            },
            "explanation_config": explanation_config.to_dict(),
            "importance_scores": {}
        }
        # Generate SHAP explanations
        if explanation_config.shap.enabled:
            print("Computing SHAP importances...")
            shap_results = self._generate_shap_explanation(
                perturbations_combined,
                scores_combined,
                neighborhood_combined,
                perturbations_test,
                scores_test,
                shap_config=explanation_config.shap
            )
            # Compute AUC for SHAP
            shap_results["auc"] = self._compute_method_auc(
                shap_results, "SHAP", true_markers
            )
            output_data["importance_scores"]["SHAP"] = shap_results
        
        # Generate Linear Regression explanations
        if explanation_config.linear_regression.enabled:
            print("Computing Linear Regression importances...")
            lr_results = self._generate_lr_explanation(
                perturbations_combined,
                scores_combined,
                neighborhood_combined,
                perturbations_test,
                scores_test,
                explanation_config.linear_regression
            )
            # Compute AUC for LR
            lr_results["auc"] = self._compute_method_auc(
                lr_results, "LR", true_markers
            )
            output_data["importance_scores"]["LR"] = lr_results
        
        # Generate Random Forest explanations
        if explanation_config.random_forest.enabled:
            print("Computing Random Forest importances...")
            rf_results = self._generate_rf_explanation(
                perturbations_combined,
                scores_combined,
                neighborhood_combined,
                perturbations_test,
                scores_test,
                explanation_config.random_forest
            )
            # Compute AUC for RF
            rf_results["auc"] = self._compute_method_auc(
                rf_results, "RF", true_markers
            )
            output_data["importance_scores"]["RF"] = rf_results
        
        
        self._save_results(output_data, output_path)
        
        return output_data
    
   
    def _generate_shap_explanation(
        self,
        perturbations: List,
        scores: List,
        neighborhoods: List,
        test_perturbations: List,
        test_scores: List,
        shap_config = None
    ) -> Dict[str, Any]:
        
        """Generate SHAP-based explanation."""
        analyzer = SHAPExplainer(perturbations, scores, neighborhoods)
        importances, _ = analyzer.get_feature_importances(empty_constraint=shap_config.empty_constraint if shap_config else None)
        mse, r2, mae, y_pred = analyzer.metrics(test_perturbations, test_scores)
        return {
            "method": "Kernel SHAP",
            "values": importances.tolist() if hasattr(importances, 'tolist') else importances,
            "test_metrics": {
                "mse": float(mse),
                "r2": float(r2),
                "mae": float(mae),
                "y_true": test_scores,
                "y_pred": y_pred.tolist()
            }
        }
    
    def _generate_lr_explanation(
        self,
        perturbations: List,
        scores: List,
        neighborhoods: List,
        test_perturbations: List,
        test_scores: List,
        lr_config = None
    ) -> Dict[str, Any]:
        
        """Generate Linear Regression-based explanation."""
        analyzer = LRExplainer(perturbations, scores, neighborhoods)
        importances, _ = analyzer.get_feature_importances(weighting=lr_config.weighting, 
                                                          regularization=lr_config.regularization, 
                                                          alpha=lr_config.alpha)
        mse, r2, mae, y_pred = analyzer.metrics(test_perturbations, test_scores)
        
        return {
            "method": "Linear Regression with kernel weighting",
            "values": importances.tolist() if hasattr(importances, 'tolist') else importances,
            "test_metrics": {
                "mse": float(mse),
                "r2": float(r2),
                "mae": float(mae),
                "y_true": test_scores,
                "y_pred": y_pred.tolist()
            }
        }
    
    def _generate_rf_explanation(
        self,
        perturbations: List,
        scores: List,
        neighborhoods: List,
        test_perturbations: List,
        test_scores: List,
        rf_config
    ) -> Dict[str, Any]:
        
        """Generate Random Forest-based explanation."""
        analyzer = RFExplainer(
            perturbations,
            scores,
            neighborhoods,
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            min_samples_leaf=rf_config.min_samples_leaf,
            random_state=rf_config.random_state
        )
        importances, _ = analyzer.get_feature_importances(
            method=rf_config.importance_method
        )
        mse, r2, mae, y_pred = analyzer.metrics(test_perturbations, test_scores)
        
        return {
            "method": f"Random Forest ({rf_config.importance_method})",
            "values": importances.tolist() if hasattr(importances, 'tolist') else importances,
            "test_metrics": {
                "mse": float(mse),
                "r2": float(r2),
                "mae": float(mae),
                "y_true": test_scores,
                "y_pred": y_pred.tolist()
            },
            "config": {
                "n_estimators": rf_config.n_estimators,
                "max_depth": rf_config.max_depth,
                "min_samples_split": rf_config.min_samples_split,
                "min_samples_leaf": rf_config.min_samples_leaf,
                "importance_method": rf_config.importance_method,
            }
        }
        
    def _save_attributions_results(self, feature_dim,
                                   attribution_results, 
                                   attribution_results_new, 
                                   method, 
                                   method_results):
        
        old_results = attribution_results[method]["values"] if method in attribution_results else [0] * feature_dim
        attribution_results_new[method] = method_results
        attribution_results_new[method]["values"] = [
                x + y
                for x, y in zip(
                    attribution_results_new[method]["values"],
                    old_results
                )
            ]
        
    def clean_gradients(self, model, sample_audio_ig=None):
        model.zero_grad()
        if sample_audio_ig is not None and sample_audio_ig.grad is not None:
            sample_audio_ig.grad.zero_()
   
    # Utility Methods
    
    def _save_results(self, output_data: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    def _compute_method_auc(
        self,
        method_results: Dict[str, Any],
        method_name: str,
        true_markers: List[List[float]]
    ) -> Optional[float]:
        """Compute AUC for a given explanation method.
        
        Args:
            method_results: Dictionary containing 'values' key with importance scores
            method_name: Name of the explanation method (SHAP, LR, RF, IG, etc.)
            true_markers: Ground truth temporal markers [[start, end], ...]
            
        Returns:
            AUC value if computation successful, None otherwise
        """
        try:
            if 'values' not in method_results:
                return None
            
            importance_scores = np.array(method_results['values'])
            
            # Create AUC evaluator with default parameters
            evaluator = AUCEvaluator(
                segment_length_ms=100.0,
                intersection_threshold=0.09,
                step_size=0.1
            )
            
            # Compute AUC
            auc_result = evaluator.compute_auc(
                importance_scores=importance_scores,
                ground_truth_markers=true_markers
            )
            
            if auc_result is not None:
                roc_auc, _ = auc_result
                print(f"  {method_name} AUC: {roc_auc:.4f}")
                return float(roc_auc)
            
        except Exception as e:
            print(f"  Warning: Could not compute AUC for {method_name}: {e}")
        
        return None
    
    @staticmethod
    def _to_list_safe(tensor: torch.Tensor) -> List[float]:
        """Safely convert tensor to list, handling single elements."""
        sq_tensor = tensor.cpu().squeeze()
        if sq_tensor.ndim == 0:
            return [sq_tensor.item()]
        return sq_tensor.tolist()
