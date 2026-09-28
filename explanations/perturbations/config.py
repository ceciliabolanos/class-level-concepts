import numpy as np
from scipy.spatial.distance import euclidean, cosine
from typing import Dict, Any
import json
from pathlib import Path

class MaskingConfig():
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_path = Path(path) / f"{self.experiment_name}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_experiment_name(self) -> str:
        """Generate descriptive name based on configuration"""
        if self.experiment_name:
            return self.experiment_name
        return self._generate_name()
    
    def _generate_name(self) -> str:
        """Override in subclasses to generate automatic names"""
        return "default_experiment"
        
class TemporalMaskingConfig(MaskingConfig):
    """Configuration for temporal masking parameters."""

    def __init__(
        self,
        mask_percentages: list = [0.4, 0.2, 0.3],
        window_sizes: list = [4, 2, 3],
        num_samples_per_batch: int = 3000,
        segment_length: int = 100,
        function: str = 'euclidean',
        mask_type: str = 'noise',
        std_noise: float = None,
        experiment_name: str = None
    ):
        super().__init__(experiment_name)
        self.mask_percentages = mask_percentages
        self.window_sizes = window_sizes
        self.num_samples = num_samples_per_batch
        self.segment_length = segment_length
        self.function = function
        self.mask_type = mask_type
        self.std_noise = std_noise
    
    def _generate_name(self) -> str:
        """Generate automatic name based on configuration"""
        mask_pct_str = "_".join([str(int(p*100)) for p in self.mask_percentages])
        win_str = "_".join(map(str, self.window_sizes))
        noise_str = f"_std{self.std_noise}" if self.std_noise else ""
        return f"temporal_{self.mask_type}_mp{mask_pct_str}_ws{win_str}_sl{self.segment_length}_ns{self.num_samples}{noise_str}"

    def masking(self, input, output, start, end):
        """Dispatcher"""

        if self.mask_type == "zeros":
            return self._masking_zeros(output, start, end)
        elif self.mask_type == "soft_zeros":
            return self._masking_soft_zeros_hann(output, start, end)
        elif self.mask_type == "noise":
            return self._masking_noise(input, output, start, end)
        elif self.mask_type == "mv_noise":
            return self._masking_mv_noise(input, output, start, end)
        elif self.mask_type == "sum_mv_noise":
            return self._masking_sum_mv_noise(output, start, end)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def _masking_zeros(self, output, start, end):
        """Reemplaza con ceros (silencio total)."""
        output[start:end] = 0
        return output
    
    def _masking_soft_zeros_hann(self, output, start, end, fade_width=None):
        """
        Alternativa: usa ventana de Hann para una atenuación más suave.
        """
        length = end - start
        
        if fade_width is None:
            fade_width = max(1, int(length * 0.1))
        else:
            fade_width = min(fade_width, length // 2)
        
        # Crear máscara con ventana de Hann en los extremos
        mask = np.ones(length)
        
        if fade_width > 0:
            # Usar la mitad izquierda de una ventana de Hann
            hann_window = np.hanning(fade_width * 2)
            mask[:fade_width] = hann_window[:fade_width]
            mask[-fade_width:] = hann_window[fade_width:]
        
        output[start:end] *= mask
        
        return output

    def _masking_noise(self, input, output, start, end):
        """Reemplaza con ruido gaussiano basado en energía global."""
        energy = input**2
        energy_p95 = np.percentile(energy, 95)
        noise_std = self.std_noise * energy_p95
        output[start:end] = np.random.normal(0, np.sqrt(noise_std), end - start)
        # noise_std = np.random.uni form(0.1 * self.std, self.std) # PAPER CODE
        # output[start:end] = np.random.normal(np.mean(self.input), noise_std, end - start) # PAPER CODE
        return output

    def _masking_mv_noise(self, input, output, start, end):
        """Reemplaza con ruido gaussiano adaptado a la energía local del segmento."""
        segment = input[start:end]
        energy_mv = np.percentile(segment**2, 95)
        noise_std = self.std_noise * energy_mv
        output[start:end] = np.random.normal(0, np.sqrt(noise_std), end - start)
        return output

    def _masking_sum_mv_noise(self, input, output, start, end):
        """Suma ruido gaussiano a la señal original (preserva señal)."""
        segment = input[start:end]
        energy_mv = np.sum(segment**2)
        noise_std = self.std_noise * energy_mv
        output[start:end] = segment + np.random.normal(0, np.sqrt(noise_std), end - start)
        return output

    def compute_similarity(self, input, temp):
        similarity_functions = {
            "euclidean": euclidean,
            "cosine": cosine,
        }
        
        similarity_function = similarity_functions.get(self.function)

        return similarity_function(input, temp)

# PREDEFINED EXPERIMENT CONFIGURATIONS

# Temporal Masking Experiments
TEMPORAL_CONFIGS = {
    "audioset_dog_noise": TemporalMaskingConfig(
        mask_percentages=[0.2, 0.4, 0.3],
        window_sizes=[1, 5, 3],
        num_samples_per_batch=512,
        segment_length=100,
        function='euclidean',
        mask_type='noise',
        std_noise=0.11,
        experiment_name="audioset_dog_noise"
    ),
    "audioset_dog_zeros": TemporalMaskingConfig(
        mask_percentages=[0.2, 0.4, 0.3],
        window_sizes=[1, 5, 3],
        num_samples_per_batch=512,
        segment_length=100,
        function='euclidean',
        mask_type='zeros',
        std_noise=None,
        experiment_name="audioset_dog_zeros"
    )
}


def get_temporal_config(exp_name: str) -> TemporalMaskingConfig:
    """Get temporal masking configuration by experiment name"""
    if exp_name not in TEMPORAL_CONFIGS:
        raise ValueError(f"Unknown temporal experiment: {exp_name}. Available: {list(TEMPORAL_CONFIGS.keys())}")
    return TEMPORAL_CONFIGS[exp_name]


def list_available_configs():
    """List all available experiment configurations"""
    print("Available Temporal Configs:")
    for name, config in TEMPORAL_CONFIGS.items():
        print(f"  - {name}: {config._generate_name()}")
