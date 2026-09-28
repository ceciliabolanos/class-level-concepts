import torch
import librosa
import numpy as np
from model.base_model import Model, UpstreamDownstreamModel
from model.utils import compute_log_odds
import ast
from pathlib import Path

class AudiosetModel(Model):
    def __init__(self, audio_path, id_to_explain: int, path):
        self.id_to_explain = id_to_explain
        self.audio_path = audio_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UpstreamDownstreamModel(upstream='mae_ast_patch', 
                                   num_layers=12, 
                                   num_classes=5,
                                   hidden_sizes=[256])

        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_predict_fn(self):
        def predict_fn(wav_array):
            if not isinstance(wav_array, list):
                wav_array = [wav_array]
            
            inputs = {
                'wav': torch.stack([torch.from_numpy(audio).float() for audio in wav_array]).to(self.device),
                'wav_lens': torch.tensor([len(audio) for audio in wav_array]).to(self.device)
            }
        
            with torch.no_grad():
                logits = self.model(inputs)
                
            logodds = compute_log_odds(logits.cpu().tolist())
            return logodds.tolist()
        
        return predict_fn
    
    def process_input(self):
        _sqf_path, rel_path = ast.literal_eval(self.audio_path)
        full_path = str(Path('/') / rel_path)

        x, fs = librosa.core.load(full_path, sr=16000)
        x = x.astype(np.float32)
        
        xin = {
            'wav': torch.from_numpy(x)[None, :],
            'wav_lens': torch.tensor([x.shape[0]])
        }
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in xin.items()}
            logits = self.model(inputs)
        logodds = compute_log_odds(logits.cpu().tolist())

        return x, logodds[0]
    