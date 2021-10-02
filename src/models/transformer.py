
from tsai.imports import *
from tsai.models.layers import *
from tsai.models.utils import *
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(Module):
    def __init__(self, device, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.

        Input shape: bs x seq x nvar
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        self.norm = Permute(0, 2, 1)
        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout,
                                                activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.norm(x) # bs x seq_len x nvars -> bs x nvars x seq_len
        x = self.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = self.transpose(x)  # seq_len x bs x d_model -> bs x seq_len x d_model
        x = self.max(x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x