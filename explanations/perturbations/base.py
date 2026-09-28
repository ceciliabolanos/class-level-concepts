import numpy as np
import os 
import random
from abc import ABC, abstractmethod
from typing import Any
from .config import MaskingConfig

class BaseDataGenerator(ABC):
    def __init__(self, 
                 config: MaskingConfig = None,
                 input: Any = None, 
                 predict_fn: Any = None,
                 path: str = None):
        """Base class for data perturbation generators."""
        self.config = config
        self.predict_fn = predict_fn
        self.input = input
        self.path = path

    @abstractmethod
    def preprocess_features(self) -> Any:
        pass
        
    @abstractmethod
    def create_perturbed_input(self) -> Any:
        """
        Given a list of 1s and 0s we generate the input masked 
        (it can be the waveform masked or the inputs_id masked).
        """
        pass

    def generate(self):
        scores, perturbations, neighborhood = self._generate_perturbations()
    
        scores_array = np.array(scores)
        perturbations_array = np.array(perturbations)
        neighborhood_array = np.array(neighborhood)
        score_real = self.predict_fn([self.input])[0]
    
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        np.savez_compressed(
            self.path,
            scores=scores_array,
            neighborhood=neighborhood_array,
            score_real=score_real,
            perturbations=perturbations_array
        )


    def time_perturbations(self, n_components, mask_percentage, window_size):
        """
        Generate masked combinations, the only thing important in here is to defined how
        many segments are we going to masked.
        Args:
            n_components (int): Total number of components
        Returns:
            list: Array of 1s and 0s representing the masking pattern
        """
        result = np.ones(n_components, dtype=int)
        
        # Calculate target number of components to mask
        target_masked = int(np.ceil(n_components * mask_percentage))
        total_masked = 0
        
        # Generate initial set of random positions
        possible_positions = list(range(n_components + 1))
        selected_positions = []
        
        while total_masked < target_masked and possible_positions:
            start_pos = random.choice(possible_positions)
            possible_positions.remove(start_pos)
            
            effective_mask = 0
            for i in range(start_pos, min(start_pos + window_size, n_components)):
                if result[i] == 1:
                    effective_mask += 1
            
            if total_masked + effective_mask > target_masked:
                continue
                
            result[start_pos:start_pos + window_size] = 0
            total_masked += effective_mask
            selected_positions.append(start_pos)
        
        if total_masked < target_masked:
            remaining = target_masked - total_masked
            while remaining > 0:
                for i in range(n_components):
                    if remaining <= 0:
                        break
                    if result[i] == 1:  # Expand around existing masked regions
                        if i > 0 and result[i - 1] == 0:
                            result[i - 1] = 1
                            remaining -= 1
                        if i < n_components - 1 and result[i + 1] == 0 and remaining > 0:
                            result[i + 1] = 1
                            remaining -= 1

        return result.tolist()

    def generate_time_combinations(self, n_components, num_samples):
        """
        Generate multiple masked combinations.
        Args:
            n_components (int): Total number of components
            num_samples (int): Number of combinations to generate        
        Returns:
            list: List of masked combinations
        """
        combinations = []
        
        param_combinations = [
            (mask_pct, window_sz) 
            for mask_pct in self.config.mask_percentages 
            for window_sz in self.config.window_sizes
        ]
    
        for _ in range(num_samples):
            chosen = random.choice(param_combinations) 
            mask_percentage, window_size = chosen
            combination = self.time_perturbations(n_components, mask_percentage, window_size)
            combinations.append(combination)
    
        random.shuffle(combinations)

        return combinations