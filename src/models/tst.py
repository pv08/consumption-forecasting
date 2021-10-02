from tsai.imports import *
from tsai.utils import *
from tsai.models.layers import *
from tsai.models.utils import *
from src.models.transformers_blocks.encoder import _TSTEncoder

class TSTModel(Module):
    def __init__(self, device, c_in: int, c_out: int, seq_len: int, max_seq_len: Optional[int] = None,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16, d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256, res_dropout: float = 0.1, act: str = "gelu", fc_dropout: float = 0.,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            res_dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.permute = Permute(0, 2, 1)
        self.c_out, self.seq_len = c_out, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len:  # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding), Conv1d(c_in, d_model, kernel_size=tr_factor, stride=tr_factor))
            pv(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n',
               verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)  # Eq 2
            pv(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        else:
            self.W_P = nn.Linear(c_in, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        W_pos = torch.zeros((q_len, d_model), device=default_device())
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.res_dropout = nn.Dropout(res_dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                   activation=act, n_layers=n_layers)
        self.flatten = Flatten()

        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, fc_dropout=fc_dropout, y_range=y_range)
        self.to(device)

    def create_head(self, nf, c_out, fc_dropout=0., y_range=None, **kwargs):
        layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # x: [bs x nvars x q_len]
        x = self.permute(x) #bs x seq_len x features -> #bs x nvars x len
        # Input encoding
        if self.new_q_len:
            u = self.W_P(x).transpose(2,
                                      1)  # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else:
            u = self.W_P(x.transpose(2,
                                     1))  # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.res_dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)  # z: [bs x q_len x d_model]
        if self.flatten is not None:
            z = self.flatten(z)  # z: [bs x q_len * d_model]
        else:
            z = z.transpose(2, 1).contiguous()  # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)  # output: [bs x c_out]