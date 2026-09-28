
from .config import MaskingConfig
from .base import BaseDataGenerator
import numpy as np
from typing import Optional, Callable
import json
from pathlib import Path

class TemporalPerturbations(BaseDataGenerator):
    """Generate time-localized perturbations"""
    def __init__(self,
                audio: np.ndarray,
                sample_rate: int,
                config: MaskingConfig,
                predict_fn: Optional[Callable] = None, 
                path: str = None):
        
        super().__init__(config=config,
                        input=audio,
                        predict_fn=predict_fn,
                        path=path)
        self.sr = sample_rate

    def preprocess_features(self):
        """
        Split signal into segments (the features)  
        """
        S = self.input
        W = [] 
        L = int((self.config.segment_length/1000) * self.sr)
        
        for start in range(0, len(S) - L + 1, L):
            end = start + L
            if end <= len(S):
                W.append(S[start:end])

        last_start = len(W) * L 
        if last_start < len(S):
            last_segment = S[last_start:]
            W.append(last_segment)
                
        return W

    def create_perturbed_input(self, row):
        W = self.preprocess_features()
        L = int((self.config.segment_length / 1000) * self.sr)
        
        output = np.copy(self.input)

        for i, (segment, use) in enumerate(zip(W, row)):
            start = i * L
            if not use:  # Apply masking
                end = start + L
                if i == len(W) - 1:  # Handle last segment to avoid index overflow
                    end = len(output)
                output = self.config.masking(self.input, output, start, end)
        return output

    def _generate_perturbations(self):
        n_components = len(self.preprocess_features())

        perturbations = self.generate_time_combinations(n_components=n_components, 
                                          num_samples=self.config.num_samples)
    
        scores, neighborhood = self.forward(perturbations=perturbations)

        return scores, perturbations, neighborhood
    
    def forward(self, perturbations=None):
        scores = []
        inputs_perturb = []
        neighborhood = []
        
        for row in perturbations:
            temp = self.create_perturbed_input(row)
            inputs_perturb.append(temp)
            neighborhood.append(self.config.compute_similarity(self.input, temp))
            
        preds = self.predict_fn(inputs_perturb)
        scores.extend(preds)
              
        return scores, neighborhood