from tsai.imports import *
from tsai.utils import *
from tsai.models.layers import *
from tsai.models.utils import *


class _ScaledDotProductAttention(Module):
    def __init__(self, d_k: int): self.d_k = d_k

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)  # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Mask (optional)
        if mask is not None: scores.masked_fill_(mask, -1e9)

        # SoftMax
        attn = F.softmax(scores, dim=-1)  # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)  # context: [bs x n_heads x q_len x d_v]

        return context, attn