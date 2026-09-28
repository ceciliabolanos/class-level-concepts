import torch
import pytorch_lightning as pl
from s3prl.nn import S3PRLUpstream
import torchmetrics

class Model:
    def __init__(self, audio_path, id_to_explain: int):
        raise NotImplementedError
    
    def get_predict_fn(self):
        raise NotImplementedError

    def process_input(self):
        raise NotImplementedError

class UpstreamDownstreamModel(pl.LightningModule):
    def __init__(self, upstream, num_layers=13, hidden_sizes=[128], num_classes=4, downstream='mlp', lstm_size=None):
        super().__init__()
        self.upstream = S3PRLUpstream(upstream)
        self.num_features = self.upstream.hidden_sizes[-1]
        self.upstream.eval()
        self.layer_weights = torch.nn.Parameter(torch.randn(num_layers))
        self.downstream_type = downstream
        self.num_classes = num_classes
        if downstream == 'mlp':
            mlp_ch = [self.num_features] + hidden_sizes + [num_classes]
            mlp_layers = [torch.nn.Sequential(torch.nn.Linear(chi, cho), torch.nn.ReLU()) for chi, cho in zip(mlp_ch[:-2],mlp_ch[1:-1])]
            mlp_layers += [torch.nn.Linear(mlp_ch[-2], mlp_ch[-1])]
            self.downstream = torch.nn.Sequential(*mlp_layers)
        elif downstream == 'lstm':
            self.downstream_lstm = torch.nn.LSTM(self.num_features, lstm_size, batch_first=True)
            mlp_ch = [lstm_size] + hidden_sizes + [num_classes]
            mlp_layers = [torch.nn.Sequential(torch.nn.Linear(chi, cho), torch.nn.ReLU()) for chi, cho in zip(mlp_ch[:-2],mlp_ch[1:-1])]
            mlp_layers += [torch.nn.Linear(mlp_ch[-2], mlp_ch[-1])]
            self.downstream_mlp = torch.nn.Sequential(*mlp_layers)
            
        if num_classes>1:
            self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
            self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        else:
            self.train_acc = torchmetrics.Accuracy(task='binary')
            self.val_acc = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        x['upstream_outs'], x['upstream_lens'] = self.upstream(x['wav'], x['wav_lens'])
        upstream_outs = torch.stack(x['upstream_outs'])
        layer_w = torch.softmax(self.layer_weights, dim=0)[:, None, None, None]
        upstream_pooled = torch.sum(layer_w*upstream_outs, dim=0)
        if self.downstream_type == 'mlp':
            padding_mask = torch.arange(0, upstream_pooled.shape[1], device=upstream_outs.device)[None,:] < x['upstream_lens'][0][:, None]
            upstream_pooled = torch.sum(upstream_pooled * padding_mask[:,:,None], dim=1)/torch.sum(padding_mask[:,:,None], dim=1)
            x['pooled_upstream'] = upstream_pooled
            x['padding_mask'] = padding_mask
            return self.downstream(x['pooled_upstream'])
        elif self.downstream_type == 'lstm':
            lstm_in = torch.nn.utils.rnn.pack_padded_sequence(upstream_pooled, 
                                                              x['upstream_lens'][0].to(device='cpu', dtype=torch.int64),
                                                              batch_first=True,
                                                              enforce_sorted=False)
            lstm_out, (hn, cn) = self.downstream_lstm(lstm_in)
            return self.downstream_mlp(hn[0])
        else:
            raise ValueError('Unknown downstream_type {}'.format(self.downstream_type))