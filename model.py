from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.api.base_model import AlgoBaseModel, ModelOutput
from torch.utils.consts import Mode
from torch import logger
from torch import Conf 

from lazydecoder import LazyDecoder, GBPOTrainer


class GRM(AlgoBaseModel):
    def __init__(self, FLAGS, embedding_group=None, **kwargs):
        if embedding_group is None:
            embedding_group = {}

        super().__init__(embedding_group=embedding_group)
        self.FLAGS = FLAGS

        self.decoder = LazyDecoder(
            vocab_size=FLAGS.vocab_size,
            d_model=getattr(FLAGS, "d_model", 768),
            n_layers=getattr(FLAGS, "n_layers", 4),
            n_heads_q=getattr(FLAGS, "n_heads_q", 12),
            gkv=getattr(FLAGS, "gkv", 3),
            d_ff=getattr(FLAGS, "d_ff", 2048),
            d_ctx_in=getattr(FLAGS, "d_ctx_in", 256),
            lkv=getattr(FLAGS, "lkv", 1),
            skv=getattr(FLAGS, "skv", 1),
            attn_drop=getattr(FLAGS, "attn_drop", 0.0),
            resid_drop=getattr(FLAGS, "resid_drop", 0.0),
        )

        self.fc_static = nn.LazyLinear(256)
        self.fc_short = nn.LazyLinear(256)
        self.fc_long = nn.LazyLinear(256)


        self.gbpo = GBPOTrainer(
            model=self.decoder,
            lambda_rl=getattr(FLAGS, "lambda_rl", 0.1),
            clip_ratio=getattr(FLAGS, "clip_ratio", 0.2),
        )


        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.decoder.pad_id,
            reduction="mean"
        )


    def forward(
        self,
        features,
        embeddings,
        embeddings_cnt,
        mode: Mode,
        **kwargs,
    ):
        group_static: Dict[str, torch.Tensor] = {}
        group_short: Dict[str, torch.Tensor] = {}
        group_long: Dict[str, torch.Tensor] = {}
        image_sid: Dict[str, torch.Tensor] = {}
        title_sid: Dict[str, torch.Tensor] = {}

        for tbl, emb_info in embeddings.items():
            if isinstance(emb_info, dict):
                emb = emb_info["tensor"]
                gid = emb_info.get("group_id", 0)
            else:
                emb = emb_info
                gid = getattr(emb, "group_id", 0)

            emb = emb.reshape(emb.shape[0], -1)

            if gid == 1:
                group_static[tbl] = emb
            elif gid == 2:
                group_short[tbl] = emb
            elif gid == 3:
                group_long[tbl] = emb
            elif "image" in tbl:
                image_sid[tbl] = emb
            elif "title" in tbl:
                title_sid[tbl] = emb

        def cat_or_none(d: Dict[str, torch.Tensor]):
            return torch.cat(list(d.values()), dim=-1) if d else None

        static_input = cat_or_none(group_static)  
        short_input  = cat_or_none(group_short)   
        long_input   = cat_or_none(group_long)    
        image_ids    = cat_or_none(image_sid)     
        title_ids    = cat_or_none(title_sid)    

        if static_input is None:
            raise ValueError("[GRM] static_input is None, please check group_id of embeddings.")

        static_hidden = F.relu(self.fc_static(static_input)) 
        short_hidden  = F.relu(self.fc_short(short_input)) if short_input is not None else torch.zeros_like(static_hidden)
        long_hidden   = F.relu(self.fc_long(long_input)) if long_input is not None else torch.zeros_like(static_hidden)

        B = static_hidden.size(0)
        device = static_hidden.device

        user_static = static_hidden.unsqueeze(1)
        short_term  = short_hidden.unsqueeze(1)
        long_term   = long_hidden.unsqueeze(1)

        bos = torch.full((B, 1), self.decoder.bos_id, dtype=torch.long, device=device)

        if image_ids is None:
            image_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            image_ids = image_ids.long()
            if image_ids.dim() == 2 and image_ids.size(1) > 1:
                image_ids = image_ids[:, :1]

        if title_ids is None:
            title_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            title_ids = title_ids.long()
            if title_ids.dim() == 2 and title_ids.size(1) > 1:
                title_ids = title_ids[:, :1]

        target_ids = torch.cat([bos, image_ids, title_ids], dim=1)  # (B, T=3)

        if mode == Mode.PREDICT:
            self.decoder.eval()
            with torch.no_grad():
                out = self.decoder(
                    target_ids,
                    user_static,
                    short_term,
                    long_term,
                    return_hidden=True
                )
                hidden = out["hidden"] 

                image_vec = hidden[:, -2, :]  
                title_vec = hidden[:, -1, :]   

                return ModelOutput(
                    logits=None,
                    predictions={
                        "creative/image_vec": image_vec,
                        "creative/title_vec": title_vec,
                        "item/item_vec": image_vec,
                    },
                    extra_outputs=None,
                )

        out = self.decoder(
            target_ids,
            user_static,
            short_term,
            long_term,
            return_hidden=False
        )
        logits = out["logits"]  

        return ModelOutput(
            logits={"decoder_logits": logits},  
            predictions=None,
            extra_outputs={
                "target_ids": target_ids,
                "user_static": user_static,
                "short_term": short_term,
                "long_term": long_term,
            }
        )

    def init_loss(self):
        pass

    def loss(
        self,
        logits,
        labels,
        weights,
        extra_outputs,
        **kwargs,
    ):

        decoder_logits = logits["decoder_logits"]        
        target_ids = extra_outputs["target_ids"]         

        ce_loss = self.ce_loss_fn(
            decoder_logits.view(-1, decoder_logits.size(-1)),
            target_ids.view(-1)
        )

        if getattr(self.FLAGS, "use_gbpo", False):
            with torch.no_grad():
                old_logits = self.decoder(
                    target_ids,
                    extra_outputs["user_static"],
                    extra_outputs["short_term"],
                    extra_outputs["long_term"]
                )["logits"]

            rewards = labels.get(
                "reward",
                torch.ones_like(target_ids, dtype=torch.float32)
            )

            gbpo_loss = self.gbpo.compute_gbpo_loss(
                new_logits=decoder_logits,
                old_logits=old_logits,
                rewards=rewards
            )

            total_loss = ce_loss + self.FLAGS.lambda_rl * gbpo_loss
        else:
            total_loss = ce_loss

        return {"decoder_logits": total_loss}

    def get_optimizer(self):
        params = []
        params += list(self.decoder.parameters())
        params += list(self.fc_static.parameters())
        params += list(self.fc_short.parameters())
        params += list(self.fc_long.parameters())

        lr = getattr(self.FLAGS, "learning_rate", 2e-4)

        return optim.Adam(
            params,
            lr=lr,
            betas=(0.9, 0.9999),
            eps=1e-8,
            weight_decay=0.0,
        )

    def backward(
        self,
        loss,
        extra_outputs,
        **kwargs,
    ):
        total_loss = torch.stack(list(loss.values())).sum()
        total_loss.backward()
        clip_value = getattr(self.FLAGS, "clip_grad_norm_value", None)
        if clip_value is not None and clip_value > 0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value) 