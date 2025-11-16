import numpy as np
import torch
import torch.nn.functional as F

from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from torch import nn
from torch import Tensor
from torch.nested import Tensor as NestedTensor
from typing import Optional
from typing import Union

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return np.reshape(t, (-1, *([1] * (x.ndim - 1))))

class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods using numpy."""

    def __init__(self, sigma=0.0):
        """Initialize the ConditionalFlowMatcher class with hyper-parameter sigma."""
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        del t  # Unused variable t in the context of this function
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + epsilon * sigma_t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field ut(x1|x0) = x1 - x0."""
        del t, xt  # Unused variables
        return x1 - x0

    def sample_noise_like(self, x):
        """Sample noise from a normal distribution N(0, 1) with the same shape as x."""
        return np.random.randn(*x.shape)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """Compute the sample xt and the conditional vector field ut."""
        if t is None:
            t = np.random.rand(x0.shape[0])
        assert len(t) == x0.shape[0], "t must have the same batch size dimension as x0"
        
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function."""
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)
    
torch.backends.cuda.enable_flash_sdp(True)

AttentionInput = Union[Tensor, NestedTensor]


class KVCache(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert len(dim) == 3, "Cache only supports 3d tensors"
        self.register_buffer("k_cache", torch.zeros(*dim, requires_grad=False))
        self.register_buffer("v_cache", torch.zeros(*dim, requires_grad=False))
        self.dim = dim
        
        self._reset_limits()
        self.is_empty = True
    
    def _reset_limits(self):
        self.cache_limits = [0 for _ in self.dim]
        self.next_seq_pos = None
    
    def reset(self):
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)
        
        self._reset_limits()
        self.is_empty = True
    
    @property
    def device(self):
        return self.k_cache.device
    
    @property
    def keys(self):
        B, N, D = self.cache_limits
        return self.k_cache[:B, :N, :D]
    
    @property
    def values(self):
        B, N, D = self.cache_limits
        return self.v_cache[:B, :N, :D]
    
    @property
    def seq_lengths(self):
        if self.is_empty:
            return 0
        return self.next_seq_pos
    
    @torch.no_grad
    def store(self, keys: Tensor, values: Tensor, mask: Tensor) -> None:
        B, N = mask.shape
        self.k_cache[:B, :N, :][mask] = keys.detach()[:, :]
        self.v_cache[:B, :N, :][mask] = values.detach()[:, :]

        self.cache_limits = [B, N, self.dim[-1]]
        self.next_seq_pos = mask.sum(axis=1).unsqueeze(-1)
        self.is_empty = False
    
    @torch.no_grad
    def append_column(self, keys: Tensor, values: Tensor) -> None:
        B, N, D = self.cache_limits

        row_idx = torch.arange(B, device=self.k_cache.device)
        self.k_cache[:B, :][row_idx, self.next_seq_pos] = keys.detach()[:, :]
        self.v_cache[:B, :][row_idx, self.next_seq_pos] = values.detach()[:, :]

        max_pos_appended = self.next_seq_pos.max()
        if max_pos_appended >= N:
            self.cache_limits[1] = max_pos_appended + 1
        self.next_seq_pos += 1
    
    @torch.no_grad
    @torch.compiler.disable
    def as_jagged(self):
        keys_jagged = padded_to_jagged_tensor(self.keys, lengths=self.seq_lengths.squeeze(), max_len=self.keys.shape[1])
        values_jagged = padded_to_jagged_tensor(self.values, lengths=self.seq_lengths.squeeze(), max_len=self.values.shape[1])
        return keys_jagged, values_jagged

    @torch.no_grad
    def apply(self, fn) -> None:
        B, N, D = self.cache_limits
        k_transformed, v_transformed = fn(self.k_cache[:B, :N, :D]), fn(self.v_cache[:B, :N, :D])
        next_seq_pos_transformed = fn(self.next_seq_pos)
        B, N, D = k_transformed.shape

        self.reset()
        self.k_cache[:B, :N, :D] = k_transformed
        self.v_cache[:B, :N, :D] = v_transformed
        self.next_seq_pos = next_seq_pos_transformed
        self.cache_limits = [B, N, D]
        self.is_empty = False


class Attend(nn.Module):
    def __init__(self, d_out, num_heads, head_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = dropout
    
    def jagged_forward(self, qu: NestedTensor, ke: NestedTensor, va: NestedTensor, is_causal: bool) -> NestedTensor:
        queries = qu.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = ke.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = va.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        dropout_p = 0. if not self.training else 0.0

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=dropout_p, is_causal=is_causal)
        
        context_vec = context_vec.transpose(1, 2).flatten(-2)
        return context_vec

    def forward(self, qkv: Tensor, is_causal: bool = False) -> Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=is_causal)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        cross_attn=False,
        dropout=0.0,
        qkv_bias=False,
        enable_kv_cache=False
    ) -> None:
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
        assert not enable_kv_cache, "KV Cache currently not supported"

        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.enable_kv_cache = enable_kv_cache

        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
    
        self.proj = nn.Linear(d_out, d_out, bias=False)

        self.attend = Attend(self.d_out, self.num_heads, self.head_dim, dropout=False)

        self._kv_cache = KVCache((2560, 80, 384)) if enable_kv_cache else None # (640, 800, 64) TODO: Revisit KV Cache
    
    @property
    def kv_cache(self) -> KVCache:
        return self._kv_cache

    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[AttentionInput] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: bool = False,
        use_cache: bool = False,
    ) -> AttentionInput:
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        assert not self.cross_attn or x_kv is not None, "Found null x_kv in cross attn. layer"
        
        if self.cross_attn:
            queries = self.q(x)
            keys, values = self.kv(x_kv).chunk(2, dim=-1)
        else:
            queries, keys, values = self.qkv(x).chunk(3, dim=-1)
        
        if not self.training and use_cache and self.enable_kv_cache and self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            
            self.kv_cache.store(
                keys=jagged_to_flattened_tensor(keys), 
                values=jagged_to_flattened_tensor(values), 
                mask=padding_mask
            )
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=is_causal)

        elif not self.training and use_cache and self.enable_kv_cache and not self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape

            keys, values = jagged_to_flattened_tensor(keys), jagged_to_flattened_tensor(values)
            
            self.kv_cache.append_column(keys=keys, values=values)
            keys, values = self.kv_cache.as_jagged()
            
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=False)
        
        elif jagged:
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=is_causal)

        if not jagged:
            raise Exception("Unjagged attention currently not supported.")
            # context_vec = self.attend(qkv, is_causal=is_causal)
    
        context_vec = self.proj(context_vec)
        return context_vec

class KVCacheOpsMixin:
    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()
    
    def apply_to_kv_cache(self, fn) -> None:
        for layer in self.layers:
            layer.apply_to_kv_cache(fn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden_dims: List[int] = [1024],
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn
        self.enable_kv_cache = enable_kv_cache

        self.attention = MultiHeadAttention(
            d_in=d_in, d_out=d_out, num_heads=num_heads, cross_attn=False, dropout=dropout, qkv_bias=qkv_bias, enable_kv_cache=enable_kv_cache
        )

        self.ff = nn.Sequential(
            RMSNorm(d_out),
            MLP(
                input_dim=d_out,
                hidden_dims=mlp_hidden_dims,
                out_dim=d_out,
                dropout=dropout,
                normalize=False
            ),
            nn.Dropout(dropout)
        )

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        self.do = nn.Dropout(dropout)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadAttention(
                d_in=d_out, d_out=d_out, num_heads=num_heads, cross_attn=True, dropout=dropout, qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out)
    
    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: Optional[bool] = False
    ) -> AttentionInput:
        attn_out = x + self.attention(self.attn_norm(x), padding_mask=padding_mask, is_causal=is_causal, jagged=jagged, use_cache=not self.training and self.enable_kv_cache)
        if self.do_cross_attn:
            attn_out = attn_out + self.cross_attention(
                x=self.cross_attn_norm(attn_out), x_kv=x_kv, padding_mask=padding_mask, is_causal=False, jagged=jagged, use_cache=not self.training and self.enable_kv_cache
            )
        proj_out = attn_out + self.ff(attn_out)
        return proj_out
    
    def reset_kv_cache(self):
        self.attention.kv_cache.reset()
        if self.do_cross_attn:
            self.cross_attention.kv_cache.reset()

    def apply_to_kv_cache(self, fn):
        self.attention.kv_cache.apply(fn)
        if self.do_cross_attn:
            self.cross_attention.kv_cache.apply(fn)


class TransformerDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()

        self.do_cross_attn = do_cross_attn

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                num_heads=num_heads,
                qkv_bias=False,
                do_cross_attn=self.do_cross_attn,
                enable_kv_cache=enable_kv_cache
            ) for _ in range(n_layers)
        ])

    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        for layer in self.layers:
            x = layer(x=x, x_kv=context, padding_mask=padding_mask, is_causal=is_causal, jagged=jagged)
        return x
    
    @property
    def seq_lengths(self) -> Tensor:
        return self.layers[0].attention.kv_cache.seq_lengths


class TransformerEncoderDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()

        self.encoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=encoder_layers,
            do_cross_attn=False,
            enable_kv_cache=False
        )

        self.decoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=decoder_layers,
            do_cross_attn=True,
            enable_kv_cache=False
        )

        self.layers = [self.encoder, self.decoder]
        self.cached_enc_output = None
    
    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        if self.cached_enc_output is None:
            context = self.encoder(context, padding_mask=padding_mask, is_causal=False, context=None, jagged=jagged)
            if not self.training:
                self.cached_enc_output = context
        else:
            context = self.cached_enc_output
        out = self.decoder(x, padding_mask=None, is_causal=True, context=context, jagged=jagged)
        return out