"""
AUC Evaluator for Explanation Quality Assessment
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import roc_curve, auc


class AUCEvaluator:
    """
    Evaluates explanation quality using AUC with relaxed segmentation.
    
    This class implements the same methodology as auc_relaxed.py:
    - Divides audio into fixed-length segments
    - Labels segments based on ground truth marker intersection
    - Computes AUC by ranking importance scores
    
    Attributes:
        segment_length_ms: Length of each segment in milliseconds (default: 100)
        intersection_threshold: Minimum overlap in seconds to label as valid (default: 0.09)
        step_size: Time step between segments in seconds (default: 0.1)
    """
    
    def __init__(
        self,
        segment_length_ms: float = 100.0,
        intersection_threshold: float = 0.09,
        step_size: float = 0.1
    ):
        """
        Initialize the AUC evaluator.
        
        Args:
            segment_length_ms: Segment length in milliseconds
            intersection_threshold: Minimum intersection in seconds for valid label
            step_size: Time step between segments in seconds
        """
        self.segment_length_ms = segment_length_ms
        self.segment_length_s = segment_length_ms / 1000.0
        self.intersection_threshold = intersection_threshold
        self.step_size = step_size
    
    def time_in_segmentation(
        self,
        segment_start: float,
        ground_truth_markers: List[List[float]]
    ) -> int:
        """
        Determine if a segment intersects with ground truth markers.
        
        Returns:
            1: Valid segment (sufficient intersection)
            -1: Discard segment (partial intersection below threshold)
            0: Negative segment (no intersection)
        
        Args:
            segment_start: Start time of the segment in seconds
            ground_truth_markers: List of [start, end] marker pairs in seconds
        """
        segment_end = segment_start + self.segment_length_s
        
        for gt_start, gt_end in ground_truth_markers:
            # Calculate intersection
            intersection_start = max(segment_start, gt_start)
            intersection_end = min(segment_end, gt_end)
            intersection_length = max(0, intersection_end - intersection_start)
            
            if intersection_length > 0:
                if intersection_length >= self.intersection_threshold:
                    return 1  # Valid segment
                else:
                    return -1  # Discard segment
        
        return 0  # Negative segment
    
    def generate_segment_times(self, num_segments: int) -> np.ndarray:
        """Generate array of segment start times."""
        return np.arange(num_segments) * self.step_size
    
    def create_segmentation_vector(
        self,
        ground_truth_markers: List[List[float]],
        segment_times: np.ndarray
    ) -> np.ndarray:
        """
        Create segmentation label vector for all segments.
        
        Args:
            ground_truth_markers: List of [start, end] marker pairs
            segment_times: Array of segment start times
            
        Returns:
            Array of labels (1, -1, or 0) for each segment
        """
        labels = np.array([
            self.time_in_segmentation(t, ground_truth_markers)
            for t in segment_times
        ])
        return labels
    
    def compute_auc(
        self,
        importance_scores: np.ndarray,
        ground_truth_markers: List[List[float]]
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Compute AUC for importance scores against ground truth.
        
        Args:
            importance_scores: Array of importance values (one per segment)
            ground_truth_markers: List of [start, end] temporal markers
            
        Returns:
            Tuple of (auc_value, fpr_values) or None if computation fails
        """
        num_segments = len(importance_scores)
        segment_times = self.generate_segment_times(num_segments)
        
        # Create segmentation labels
        segmentation_labels = self.create_segmentation_vector(
            ground_truth_markers, segment_times
        )
        
        # Filter out discard segments (label = -1)
        valid_mask = segmentation_labels != -1
        filtered_labels = segmentation_labels[valid_mask]
        filtered_scores = importance_scores[valid_mask]
        
        if len(filtered_labels) == 0 or len(np.unique(filtered_labels)) < 2:
            return None
        
        # Rank transformation: highest importance -> 1, lowest -> 0
        ranked_scores = np.linspace(1, 0, len(filtered_scores))
        sorted_indices = np.argsort(-filtered_scores)  # Descending order
        ranking = np.empty_like(ranked_scores)
        ranking[sorted_indices] = ranked_scores
        
        # Compute ROC curve and AUC
        try:
            fpr, tpr, _ = roc_curve(filtered_labels, ranking)
            roc_auc = auc(fpr, tpr)
            return roc_auc, fpr
        except Exception as e:
            print(f"Error computing AUC: {e}")
            return None
