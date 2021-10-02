from tsai.imports import *
from tsai.utils import *
from tsai.models.layers import *
from tsai.models.utils import *
from src.models.transformers_blocks.multi_head_attention import _MultiHeadAttention

class _TSTEncoderLayer(Module):
    def __init__(self, q_len:int, d_model:int, n_heads:int, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, res_dropout:float=0.1,
                 activation:str="gelu"):

        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.batchnorm_attn = nn.BatchNorm1d(q_len)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(q_len)

    def forward(self, src:Tensor, mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_attn(src)      # Norm: batchnorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_ffn(src) # Norm: batchnorm

        return src

    def _get_activation_fn(self, activation):
        if activation == "relu": return nn.ReLU()
        elif activation == "gelu": return nn.GELU()
        else: return activation()
#         raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')


class _TSTEncoder(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu',
                 n_layers=1):
        self.layers = nn.ModuleList(
            [_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                              activation=activation) for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output