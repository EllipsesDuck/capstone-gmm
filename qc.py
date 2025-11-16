import torch

from data import SeqBatch
from einops import rearrange
from functools import cached_property
from modules import *
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor
from enum import Enum

torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor   # [B, D, L]
    residuals: Tensor    # [B, D, L]
    sem_ids: Tensor      # [B, L]
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor        # [B, L]
    p_unique_ids: Tensor     # scalar


class QuantizerType(str, Enum):
    RQ = "rq"
    RQ_KMEANS = "rq_kmeans"
    VQ = "vq"
    PQ = "pq" 


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        quantizer_type: str | QuantizerType = QuantizerType.RQ,
        pq_n_subspaces: int = 4,
    ) -> None:
        self._config = locals()
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        if not isinstance(quantizer_type, QuantizerType):
            quantizer_type = QuantizerType(quantizer_type)
        self.quantizer_type = quantizer_type

        self.pq_n_subspaces = pq_n_subspaces
        if self.quantizer_type == QuantizerType.PQ:
            assert embed_dim % pq_n_subspaces == 0, \
                f"embed_dim({embed_dim}) must be divisible by pq_n_subspaces({pq_n_subspaces})"
            self.pq_subspace_dim = embed_dim // pq_n_subspaces
            self.n_layers = pq_n_subspaces
        else:
            self.pq_subspace_dim = None
            self.n_layers = n_layers

        if self.quantizer_type in (QuantizerType.RQ, QuantizerType.RQ_KMEANS, QuantizerType.VQ):
            if self.quantizer_type == QuantizerType.VQ:
                self.n_layers = 1

            self.layers = nn.ModuleList(
                [
                    Quantize(
                        embed_dim=embed_dim,
                        n_embed=codebook_size,
                        forward_mode=codebook_mode,
                        do_kmeans_init=(
                            codebook_kmeans_init
                            if self.quantizer_type != QuantizerType.RQ_KMEANS
                            else True
                        ),
                        codebook_normalize=(i == 0 and codebook_normalize),
                        sim_vq=codebook_sim_vq,
                        commitment_weight=commitment_weight,
                    )
                    for i in range(self.n_layers)
                ]
            )
            self.pq_layers = None

        elif self.quantizer_type == QuantizerType.PQ:
            self.layers = None
            self.pq_layers = nn.ModuleList(
                [
                    Quantize(
                        embed_dim=self.pq_subspace_dim,
                        n_embed=codebook_size,
                        forward_mode=codebook_mode,
                        do_kmeans_init=codebook_kmeans_init,
                        codebook_normalize=codebook_normalize,
                        sim_vq=codebook_sim_vq,
                        commitment_weight=commitment_weight,
                    )
                    for _ in range(self.pq_n_subspaces)
                ]
            )
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True,
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)


    def _get_semantic_ids_rq_like(
        self, z: Tensor, gumbel_t: float
    ) -> RqVaeOutput:
        residual = z
        quantize_loss = 0.0

        embs_list = []      
        residuals_list = [] 
        sem_ids_list = []  

        for layer in self.layers:
            residuals_list.append(residual)
            quantized = layer(residual, temperature=gumbel_t)
            quantize_loss = quantize_loss + quantized.loss

            emb, ids = quantized.embeddings, quantized.ids  # [B, D], [B]
            residual = residual - emb

            embs_list.append(emb)
            sem_ids_list.append(ids)

        embs = torch.stack(embs_list, dim=0)        
        residuals = torch.stack(residuals_list, 0)
        sem_ids = torch.stack(sem_ids_list, dim=1)  
        embs = rearrange(embs, "l b d -> b d l")
        residuals = rearrange(residuals, "l b d -> b d l")

        return RqVaeOutput(
            embeddings=embs,
            residuals=residuals,
            sem_ids=sem_ids,
            quantize_loss=quantize_loss,
        )

    def _get_semantic_ids_pq(
        self, z: Tensor, gumbel_t: float
    ) -> RqVaeOutput:
        B, D = z.shape
        sub_dim = self.pq_subspace_dim
        K = self.pq_n_subspaces

        # [B, K, sub_dim]
        z_view = z.view(B, K, sub_dim)

        quantize_loss = 0.0
        embs_full_list = []      
        residuals_full_list = []  
        sem_ids_list = []         
        for k in range(K):
            z_sub = z_view[:, k, :]  # [B, sub_dim]
            quantized = self.pq_layers[k](z_sub, temperature=gumbel_t)
            quantize_loss = quantize_loss + quantized.loss

            emb_sub, ids = quantized.embeddings, quantized.ids  # [B, sub_dim], [B]

            emb_full = torch.zeros(B, D, device=z.device, dtype=z.dtype)
            start = k * sub_dim
            end = (k + 1) * sub_dim
            emb_full[:, start:end] = emb_sub

            residual_full = z - emb_full

            embs_full_list.append(emb_full)
            residuals_full_list.append(residual_full)
            sem_ids_list.append(ids)

        # [K, B, D] → [B, D, K]
        embs = torch.stack(embs_full_list, dim=0)
        residuals = torch.stack(residuals_full_list, dim=0)
        embs = rearrange(embs, "k b d -> b d k")
        residuals = rearrange(residuals, "k b d -> b d k")

        sem_ids = torch.stack(sem_ids_list, dim=1)  # [B, K]

        return RqVaeOutput(
            embeddings=embs,
            residuals=residuals,
            sem_ids=sem_ids,
            quantize_loss=quantize_loss,
        )

    def get_semantic_ids(
        self,
        x: Tensor,
        gumbel_t: float = 0.001,
    ) -> RqVaeOutput:
        z = self.encode(x)

        if self.quantizer_type == QuantizerType.PQ:
            return self._get_semantic_ids_pq(z, gumbel_t)
        else:
            return self._get_semantic_ids_rq_like(z, gumbel_t)

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch.x  # [B, input_dim]

        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals  # [B, D, L], [B, D, L]

        z_hat = embs.sum(dim=-1)

        x_hat = self.decode(z_hat)  # [B, input_dim]

        if self.n_cat_feats > 0:
            cont = l2norm(x_hat[..., :-self.n_cat_feats])
            cat = x_hat[..., -self.n_cat_feats:]
            x_hat = torch.cat([cont, cat], dim=-1)
        else:
            x_hat = l2norm(x_hat)

        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        with torch.no_grad():
            embs_norm = embs.norm(dim=1)  

            # sem_ids: [B, L]
            ids = quantized.sem_ids
            # [B, 1, L] & [1, B, L]
            ids1 = rearrange(ids, "b l -> b 1 l")
            ids2 = rearrange(ids, "b l -> 1 b l")
            same_pattern = (ids1 == ids2).all(dim=-1)  # [B, B]
            p_unique_ids = (
                (~torch.triu(same_pattern, diagonal=1))
                .all(dim=1)
                .sum()
                .float()
                / ids.shape[0]
            )

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )