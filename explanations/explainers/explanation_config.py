"""
Configuration system for explanation methods.
Defines configurations for different explainer types (SHAP, LR, RF, CNN-based methods).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import json
from pathlib import Path
from datetime import datetime

@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""
    enabled: bool = True
    empty_constraint: str = None  # kernel, tree, deep, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'empty_constraint': self.empty_constraint
        }

@dataclass
class LinearRegressionConfig:
    """Configuration for Linear Regression explainer."""
    enabled: bool = True
    weighting: bool = False
    regularization: str =None
    alpha: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'weighting': self.weighting,
            'regularization': self.regularization,
            'alpha': self.alpha
        }

@dataclass
class RandomForestConfig:
    """Configuration for Random Forest explainer."""
    enabled: bool = True
    n_estimators: int = 100
    max_depth: Optional[int] = None
    max_features: Optional[str] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    importance_method: Literal['gini', 'permutation'] = 'gini'
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            "max_features": self.max_features,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'importance_method': self.importance_method,
            'random_state': self.random_state
        }

@dataclass
class SurrogateTrainingConfig:
    """Configuration for surrogate model training (for CNN-based explanations)."""
    train_test_split: float = 0.85  # 85% train, 15% test
    validation_split: float = 0.15  # From training set, 15% for validation
    random_seed: int = 42
    compute_learning_curves: bool = True
    compute_held_out_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_test_split': self.train_test_split,
            'validation_split': self.validation_split,
            'random_seed': self.random_seed,
            'compute_learning_curves': self.compute_learning_curves,
            'compute_held_out_metrics': self.compute_held_out_metrics
        }


@dataclass
class ExplanationConfig:
    """
    Complete configuration for explanation generation.
    Combines all explainer configurations and metadata.
    """
    name: str
    description: str
    
    # Enable/disable different explainer types
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    linear_regression: LinearRegressionConfig = field(default_factory=LinearRegressionConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)

    # Metadata
    date_created: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Set creation date if not provided."""
        if self.date_created is None:
            self.date_created = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        config_dict = {
            'name': self.name,
            'description': self.description,
            'shap': self.shap.to_dict(),
            'linear_regression': self.linear_regression.to_dict(),
            'random_forest': self.random_forest.to_dict(),
            'surrogate_training': self.surrogate_training.to_dict(),
            'date_created': self.date_created,
            'notes': self.notes
        }
        
        if self.cnn_explainer is not None:
            config_dict['cnn_explainer'] = self.cnn_explainer.to_dict()
        
        return config_dict
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        config_path = Path(path) / f"{self.name}_explanation_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Explanation config saved to: {config_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplanationConfig':
        """Load configuration from dictionary."""
        # Create nested config objects
        shap = SHAPConfig(**config_dict.get('shap', {}))
        lr = LinearRegressionConfig(**config_dict.get('linear_regression', {}))
        rf = RandomForestConfig(**config_dict.get('random_forest', {}))
        surrogate = SurrogateTrainingConfig(**config_dict.get('surrogate_training', {}))
        
        return cls(
            name=config_dict['name'],
            description=config_dict['description'],
            shap=shap,
            linear_regression=lr,
            random_forest=rf,
            surrogate_training=surrogate,
            date_created=config_dict.get('date_created'),
            notes=config_dict.get('notes')
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ExplanationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingMetrics:
    """
    Store training metrics including learning curves and held-out test results.
    """
    # Learning curves (per epoch)
    batch_losses: List[List[float]] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    # Held-out test metrics
    test_mse: List[float] = field(default_factory=list)
    test_r2: List[float] = field(default_factory=list)
    test_mae: List[float] = field(default_factory=list)
    predicted_ones: List[float]  = field(default_factory=list)
    predicted_zeros: List[float]  = field(default_factory=list)
    y_true: List[List[float]] = field(default_factory=list)
    y_pred: List[List[float]] = field(default_factory=list)
    
    # Additional metadata
    num_epochs: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0
    num_parameters: Optional[int] = None
    
    def add_epoch(self, batch_loss: List[float], train_loss: float, val_loss: Optional[float] = None):
        """Add metrics for an epoch."""
        self.batch_losses.append(batch_loss)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.num_epochs += 1
    
    def set_test_metrics(self, mse: float, r2: float, mae: Optional[float] = None, 
                        predicted_ones: Optional[float] = None, predicted_zeros: Optional[float] = None, 
                        y_true: Optional[List[float]] = None, y_pred: Optional[List[float]] = None):
        """Set held-out test set metrics.""" 
        self.test_mse.append(mse)
        self.test_r2.append(r2)
        self.test_mae.append(mae)
        self.predicted_ones.append(predicted_ones)
        self.predicted_zeros.append(predicted_zeros)
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'learning_curves': {
                'batch_losses': self.batch_losses,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'num_epochs': self.num_epochs
            },
            'held_out_test': {
                'mse': self.test_mse,
                'r2': self.test_r2,
                'mae': self.test_mae,
                'predicted_ones': self.predicted_ones,
                'predicted_zeros': self.predicted_zeros,
                'y_true': self.y_true,
                'y_pred': self.y_pred,
            },
            'data_splits': {
                'num_train_samples': self.num_train_samples,
                'num_val_samples': self.num_val_samples,
                'num_test_samples': self.num_test_samples
            }
        }

EXPLANATION_CONFIGS = {
    'paper_default': ExplanationConfig(
        name='paper_default',
        description='Paper default values',
        shap=SHAPConfig(enabled=True, 
                        empty_constraint=None),
        linear_regression=LinearRegressionConfig(weighting=False,
                                                regularization=None,
                                                alpha=1.0),
        random_forest=RandomForestConfig(n_estimators=100,
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        importance_method='gini',
                                        random_state=42),
    ),
}

def get_explanation_config(name: str) -> ExplanationConfig:
    """
    Get a predefined explanation configuration by name.
    
    Args:
        name: Configuration name (e.g., 'all_methods', 'surrogate_only')
    
    Returns:
        ExplanationConfig object
    
    Raises:
        ValueError: If configuration name not found
    """
    if name not in EXPLANATION_CONFIGS:
        available = ', '.join(EXPLANATION_CONFIGS.keys())
        raise ValueError(
            f"Configuration '{name}' not found. Available configs: {available}"
        )
    return EXPLANATION_CONFIGS[name]


def list_explanation_configs() -> List[str]:
    """List all available explanation configuration names."""
    return list(EXPLANATION_CONFIGS.keys())
