from tsai.imports import *
from tsai.utils import *
from tsai.models.layers import *
from tsai.models.utils import *
from src.models.transformers_blocks.product_attention import _ScaledDotProductAttention

class _MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None):
        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s,
                                                             v_s)  # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1,
                                                            self.n_heads * self.d_v)  # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)  # context: [bs x q_len x d_model]

        return output, attn