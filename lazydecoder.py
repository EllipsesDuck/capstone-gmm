from typing import List, Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _split_heads(x, num_heads):
    B, T, D = x.shape
    assert D % num_heads == 0, f"Hidden size {D} not divisible by heads {num_heads}"
    d = D // num_heads
    return x.view(B, T, num_heads, d)


def _merge_heads(x):
    """(B, T, H, d) -> (B, T, H*d)
    """
    B, T, H, d = x.shape
    return x.contiguous().view(B, T, H * d)


def scaled_dot_product_attention(q,k,v,attn_mask,dropout_p=0.0,training=True):
    d = q.size(-1)
    # (B, H_q, T_q, d) @ (B, H_k, d, T_k) → (B, H_q, T_q, T_k)
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 3, 1)
    attn_scores = torch.matmul(q_, k_) / math.sqrt(d)

    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask  

    attn_probs = F.softmax(attn_scores, dim=-1)
    if dropout_p > 0 and training:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    # (B, H_q, T_q, T_k) @ (B, H_k, T_k, d) → (B, H_q, T_q, d)
    v_ = v.permute(0, 2, 1, 3)
    context = torch.matmul(attn_probs, v_)
    # -> (B, T_q, H_q, d)
    context = context.permute(0, 2, 1, 3)
    return context, attn_probs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.norm_qkv = RMSNorm(d_model)

    def forward(self, x, causal=True):
        # x: (B, T, D)
        B, T, D = x.shape
        x = self.norm_qkv(x)
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        # causal mask: (1, 1, T, T)
        attn_mask = None
        if causal:
            mask = torch.full((T, T), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

        ctx, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.attn_drop, training=self.training)
        out = _merge_heads(ctx)  # (B, T, D)
        out = self.out_proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out


class LazyCrossAttentionGQA(nn.Module):
    def __init__(self, d_model, n_heads_q, gkv, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert n_heads_q % gkv == 0 
        self.d_model = d_model
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_head = d_model // n_heads_q

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.norm_q = RMSNorm(d_model)

    def forward(self,x_q,k_ctx,v_ctx,attn_mask=None):
        # x_q: (B, T_q, D); k_ctx/v_ctx: (B, T_k, Gkv, d_head)
        B, Tq, D = x_q.shape
        _, Tk, Gkv, d = k_ctx.shape
        assert Gkv == self.gkv and d == self.d_head

        q = self.q_proj(self.norm_q(x_q))  # (B, Tq, D)
        q = _split_heads(q, self.n_heads_q)  # (B, Tq, Hq, d)

        repeat = self.n_heads_q // self.gkv
        k = k_ctx.repeat_interleave(repeat, dim=2)
        v = v_ctx.repeat_interleave(repeat, dim=2)

        ctx, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.attn_drop, training=self.training)
        out = _merge_heads(ctx)  # (B, Tq, D)
        out = self.out_proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop=0.0, activation="silu"):
        super().__init__()
        act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            act,
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.drop = drop
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x_in = self.norm(x)
        y = self.net(x_in)
        if self.drop > 0:
            y = F.dropout(y, p=self.drop, training=self.training)
        return y


class ContextProcessor(nn.Module):
    def __init__(self,
                 d_in,
                 d_head,
                 gkv,
                 lkv=1,
                 skv=1,
                 use_norm_k=True,
                 use_norm_v=True):
        super().__init__()
        assert skv in (1, 2)
        self.d_in = d_in
        self.d_head = d_head
        self.gkv = gkv
        self.lkv = lkv
        self.skv = skv
        self.d_context = skv * lkv * gkv * d_head

        self.proj = nn.Linear(d_in, self.d_context, bias=False)
        self.norm_k_layers = nn.ModuleList([RMSNorm(gkv * d_head) if use_norm_k else nn.Identity()
                                            for _ in range(lkv)])
        self.norm_v_layers = nn.ModuleList([RMSNorm(gkv * d_head) if use_norm_v else nn.Identity()
                                            for _ in range(lkv)])

    def forward(self,user_static,short_term,long_term):
        ctx_parts = []
        for x in (user_static, short_term, long_term):
            if x is not None:
                # x: (B, T, D_in) to (B, T, d_context)
                ctx_parts.append(self.proj(x))
        assert len(ctx_parts) > 0 
        ctx = torch.cat(ctx_parts, dim=1) if len(ctx_parts) > 1 else ctx_parts[0]

        B, Tctx, D = ctx.shape
        assert D == self.d_context

        chunk_size = self.skv * self.gkv * self.d_head
        chunks = ctx.split(chunk_size, dim=-1) 
        assert len(chunks) == self.lkv, f"期望 {self.lkv} 份，得到 {len(chunks)}"

        kv_list = []
        for l, ch in enumerate(chunks):
            # ch: (B, Tctx, skv*gkv*d_head)
            if self.skv == 1:
                k = ch  # (B, Tctx, gkv*d_head)
                k = self.norm_k_layers[l](k)
                k = k.view(B, Tctx, self.gkv, self.d_head)
                v = k
            else:
                mid = (self.gkv * self.d_head)
                k, v = ch[..., :mid], ch[..., mid:]
                k = self.norm_k_layers[l](k)
                v = self.norm_v_layers[l](v)
                k = k.view(B, Tctx, self.gkv, self.d_head)
                v = v.view(B, Tctx, self.gkv, self.d_head)
            kv_list.append((k, v))
        return kv_list


class LazyDecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads_q,
                 gkv,
                 d_ff,
                 attn_drop=0.0,
                 resid_drop=0.0):
        super().__init__()
        self.cross_attn = LazyCrossAttentionGQA(d_model, n_heads_q, gkv,
                                               attn_drop=attn_drop, proj_drop=resid_drop)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads_q,
                                               attn_drop=attn_drop, proj_drop=resid_drop)
        self.ffn = FeedForward(d_model, d_ff, drop=resid_drop)

    def forward(self,
                x,
                k_ctx,
                v_ctx,
                causal=True):
        # Cross-Attn
        x = x + self.cross_attn(x, k_ctx, v_ctx, attn_mask=None)
        # Self-Attn
        x = x + self.self_attn(x, causal=causal)
        # FFN
        x = x + self.ffn(x)
        return x


class LazyDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model = 768,
                 n_layers = 12,
                 n_heads_q = 12,
                 gkv = 3,
                 d_ff = 2048,
                 # Context Processor
                 d_ctx_in = 256,
                 lkv = 1,
                 skv = 1,
                 pad_id = 0,
                 bos_id = 1,
                 attn_drop = 0.0,
                 resid_drop = 0.0):
        super().__init__()
        assert d_model % n_heads_q == 0
        assert n_heads_q % gkv == 0

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_head = d_model // n_heads_q
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.lkv = lkv
        self.skv = skv

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.ctx_proc = ContextProcessor(
            d_in=d_ctx_in,
            d_head=self.d_head,
            gkv=gkv,
            lkv=lkv,
            skv=skv,
        )

        self.blocks = nn.ModuleList([
            LazyDecoderBlock(d_model, n_heads_q, gkv, d_ff,
                             attn_drop=attn_drop, resid_drop=resid_drop)
            for _ in range(n_layers)
        ])

    def _kv_index_for_layer(self, l):
        # l ∈ [0, Nlayer-1] → l_kv ∈ [0, Lkv-1]
        return (l * self.lkv) // self.n_layers

    def forward(self,
                target_ids,  # (B, T_gen)
                user_static,  # (B, Ns, d_ctx_in)
                short_term,  # (B, Ts, d_ctx_in)
                long_term,   # (B, Tl, d_ctx_in)
                return_hidden = False):
        kv_list = self.ctx_proc(user_static, short_term, long_term)

        x = self.tok_emb(target_ids)  # (B, T, D)

        for l, blk in enumerate(self.blocks):
            idx = self._kv_index_for_layer(l)
            k_ctx, v_ctx = kv_list[idx]
            x = blk(x, k_ctx, v_ctx, causal=True)

        h = self.out_norm(x)
        logits = self.out_proj(h)  # (B, T, vocab_size)
        out = {"logits": logits}
        if return_hidden:
            out["hidden"] = h
        return out

    @torch.no_grad()
    def step(self,
             prev_ids: torch.Tensor,     # (B, T_prev)
             user_static: Optional[torch.Tensor] = None,
             short_term: Optional[torch.Tensor] = None,
             long_term: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.forward(prev_ids, user_static, short_term, long_term, return_hidden=False)
        logits_last = out["logits"][:, -1, :]  # (B, vocab)
        return logits_last


class GBPOTrainer:
    def __init__(self, model, lambda_rl=0.1, clip_ratio=0.2):
        self.model = model
        self.lambda_rl = lambda_rl
        self.clip_ratio = clip_ratio

    def compute_supervised_loss(self, logits, targets, pad_id=0):
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=pad_id
        )
        return loss

    def compute_gbpo_loss(self,
                          new_logits,
                          old_logits,
                          rewards,
                          mask=None):
        logp_new = F.log_softmax(new_logits, dim=-1)
        logp_old = F.log_softmax(old_logits.detach(), dim=-1)

        probs_new = logp_new.exp().clamp_min(1e-9)
        probs_old = logp_old.exp().clamp_min(1e-9)
        ratio = (probs_new / probs_old).clamp(1e-3, 10)

        adv = rewards.unsqueeze(-1)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_unclipped = -ratio * adv
        loss_clipped = -clipped_ratio * adv
        loss = torch.max(loss_unclipped, loss_clipped)

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
        return loss.mean()

    def train_step(self,
                   batch,
                   optimizer,
                   use_rl=False,
                   old_logits=None,
                   rewards=None):
        target_ids = batch["target_ids"]
        user_static = batch.get("user_static", None)
        short_term = batch.get("short_term", None)
        long_term = batch.get("long_term", None)

        out = self.model(target_ids, user_static, short_term, long_term)
        logits = out["logits"]

        if not use_rl:
            loss = self.compute_supervised_loss(logits, target_ids)
        else:
            assert old_logits is not None and rewards is not None
            loss_rl = self.compute_gbpo_loss(logits, old_logits, rewards)
            loss_ce = self.compute_supervised_loss(logits, target_ids)
            loss = loss_ce + self.lambda_rl * loss_rl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()