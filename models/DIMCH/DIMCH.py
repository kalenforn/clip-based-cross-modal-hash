import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hash.hash import HashLayer
from ..base import BaseModel
from common.register import registry
from .loss.triplet_loss import TripletLoss
from einops import repeat
from .distance import SetwiseDistance

@registry.register_model("DIMCH")
class DIMCH(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/DIMCH/base.yaml"}

    def __init__(self, 
                cfg,
                outputDim=16, 
                clipPath="./ViT-B-32.pt",
                triplet_margin=0.1,
                train_num=10000,
                txt_token_size=32, 
                setDim=64, 
                dropout=0.3,
                hash_func: str="softmax",
                merge_func: str="mean",
                cls_alpha=0.7, 
                **kwags):
        super().__init__()
        # assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        self.hash_func = hash_func
        embed_dim, visual_token_size, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=True)
        self.visual_token_size = visual_token_size
        self.txt_token_size = txt_token_size
        self.cls_alpha = cls_alpha
        self.hash = HashLayer(visual_tokens=visual_token_size, txt_tokens=txt_token_size, 
                                                 feature_size=embed_dim, outputDim=outputDim, setDim=setDim, dropout=dropout, 
                                                 hash_func_=hash_func, merge_func=merge_func)
        self.output_dim = outputDim
        self.triplet_loss = TripletLoss(reduction="mean")
        self.distance = SetwiseDistance(img_set_size=setDim, txt_set_size=setDim, 
                                        denominator=cfg.distance.get("denominator", 2.0), temperature=cfg.distance.get("temperature", 16.0), 
                                        temperature_txt_scale=cfg.distance.get("temperature_txt_scale", 1.0), mode=cfg.distance.get("mode", "chamfer"))
        self.chamfer_parmeters = {
            "set_size": setDim, 
            "margin": self.cfg.chamfer.get("margin", 0.5), 
            "semi_hard_triplet": self.cfg.chamfer.get("semi_hard_triplet", False), 
            "max_violation": self.cfg.chamfer.get("max_violation", True),
            "mmd_alpha": self.cfg.chamfer.get("mmd_alpha", 0.01),
            "unif_alpha": self.cfg.chamfer.get("unif_alpha", 0.01),
            "mmd_gamma": self.cfg.chamfer.get("mmd_gamma", 0.5),
            "token_triplet_margin": self.cfg.chamfer.get("token_triplet_margin", 0.2)
        }
        self.hash_parameters = {
            "triplet_alpha": self.cfg.hash_pars.get("triplet_alpha", 1), 
            "infonce_alpha": self.cfg.hash_pars.get("infonce_alpha", 0), 
            "quan_alpha": self.cfg.hash_pars.get("quan_alpha", 0.001),
            "hash_triplet_alpha": self.cfg.hash_pars.get("hash_triplet_alpha", 0.5),
        }
        triplet_margin = self.cfg.hash_pars.get("triplet_margin", 0.3)
        self.triplet_margin = triplet_margin

    def encode_image(self, image):

        cls_token, seq_tokens, _ = self.backbone.encode_image(image)
        if seq_tokens.shape[0] != self.visual_token_size and seq_tokens.shape[1] != self.visual_token_size:
            tokens = torch.vstack([cls_token.unsqueeze(dim=0), seq_tokens])
        else:
            tokens = seq_tokens
        # print(self.visual_token_size)
        tokens = tokens.permute(1, 0, 2) if tokens.shape[1] != self.visual_token_size else tokens
        # print(tokens.shape)
        token_embeds, token_hash = self.hash.encode_img(tokens)

        return token_embeds, token_hash

    def encode_text(self, text):
        _, tokens, _, _ = self.backbone.encode_text(text)
        tokens = tokens.permute(1, 0, 2) if tokens.shape[1] != self.txt_token_size else tokens
        token_embeds, token_hash = self.hash.encode_txt(tokens)

        return token_embeds, token_hash
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000, txt_token_size=32):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        loss_type = cfg.get("loss_type", "l1")
        triplet_margin = cfg.get("triplet_margin", 0.1)
        setDim = cfg.get("setDim", 64)
        dropout = cfg.get("dropout", 0.3)
        hash_func = cfg.get("hash_func", "softmax")
        merge_func = cfg.get("merge_func", "mean")
        cls_alpha = cfg.get("cls_alpha", "0.7")
        if "softmax" in hash_func:
            output_dim *= 2 

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            loss_type=loss_type, 
            triplet_margin=triplet_margin, 
            train_num=train_num,
            txt_token_size=txt_token_size,
            setDim=setDim, 
            dropout=dropout,
            hash_func=hash_func,
            merge_func=merge_func,
            cls_alpha=cls_alpha
        )
        return model
    
    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        img_embeds, img_hash = self.encode_image(image)
        txt_embeds, txt_hash = self.encode_text(text)
        
        if return_loss:
            return self.object_function(img_hash, txt_hash, labels=labels, indexs=indexs, img_embeds=img_embeds, txt_embeds=txt_embeds)
        
        return img_embeds, img_hash, txt_embeds, txt_hash
    
    def soft_argmax_hash_loss(self, code):
        if len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)
        
        hash_loss = 1 - torch.pow(2 * code - 1, 2).mean()
        return hash_loss

    def tanh_hash_loss(self, code):
        
        hash = torch.sign(code.detach())
        return F.mse_loss(code, hash)

    def trip_loss(self, image, text, label, indexs=None, **kwags):
        assert self.triplet_loss is not None, "please initialing self.triplet_loss before computing."
        i2t_loss = self.triplet_loss(image, label, target=text, margin=self.triplet_margin)
        t2i_loss = self.triplet_loss(text, label, target=image, margin=self.triplet_margin)

        if self.hash_func == "softmax":
            quan_image_loss = self.soft_argmax_hash_loss(image)
            quan_text_loss = self.soft_argmax_hash_loss(text)
        elif self.hash_func == "tanh":
            quan_image_loss = self.tanh_hash_loss(image)
            quan_text_loss = self.tanh_hash_loss(text)

        return i2t_loss, t2i_loss, quan_image_loss, quan_text_loss
    
    
    def chamfer_similarity_loss(self, img_embeds, img_hash, txt_embeds, txt_hash, labels=None, indexs=None, **kwags):
        
        assert self.distance is not None, "chamfer similarity loss must initialize the 'self.distance' parameter before computing."
        
        def rbf_memory_efficient(x, y, gamma):
            """RBF kernel that does not cause memory shortage"""
            cdist = torch.cdist(x, y)
            return torch.exp(-gamma * cdist)

        # for Maximum Mean Discrepancy
        def mmd_rbf_loss(x, y, gamma=None, reduction='mean'):
            if gamma is None:
                gamma = 1./x.size(-1)
            if reduction=='mean':
                loss = rbf_memory_efficient(x, x, gamma).mean() - 2 * rbf_memory_efficient(x, y, gamma).mean() + rbf_memory_efficient(y, y, gamma).mean()
            else:
                loss = rbf_memory_efficient(x, x, gamma).sum() - 2 * rbf_memory_efficient(x, y, gamma).sum() + rbf_memory_efficient(y, y, gamma).sum()
            return loss

        # for token embeds diverse
        def batchwise_uniformity_loss(embs, num_embeds, t=20):
            if num_embeds == 1:
                return 0.0
            rbf = torch.exp(-t * torch.cdist(embs, embs).pow(2))
            I = torch.autograd.Variable(repeat(
                torch.triu(torch.ones(rbf.shape[1], rbf.shape[1]), diagonal=1), 
                'n d -> b n d', 
                b=rbf.shape[0]
            )).to(embs.device)
            rbf = torch.where(I == 1, rbf, torch.zeros_like(rbf))
            loss = torch.stack([r.sum() for r in rbf]) / (num_embeds * (num_embeds - 1) * 0.5)
            return loss.mean()

        ####
        # for the tokens:
        img_embeds = img_embeds.view(-1, img_embeds.shape[-1])
        txt_embeds = txt_embeds.view(-1, txt_embeds.shape[-1])
        # normalize for cosine distance
        img_embeds = F.normalize(img_embeds, dim=-1)
        txt_embeds = F.normalize(txt_embeds, dim=-1)
        loss = 0

        i2t_embed_similarity = self.distance.compute(img_embs=img_embeds, txt_embs=txt_embeds)
        t2i_embed_similarity = self.distance.compute(img_embs=txt_embeds, txt_embs=img_embeds)
        tokens_i2t_loss = self.triplet_loss(None, labels, target=None, distance=torch.clamp(1 - i2t_embed_similarity, 0), margin=self.chamfer_parmeters["token_triplet_margin"])
        tokens_t2i_loss = self.triplet_loss(None, labels, target=None, distance=torch.clamp(1 - t2i_embed_similarity, 0), margin=self.chamfer_parmeters["token_triplet_margin"])
        
        # mmd loss
        mmd_loss = mmd_rbf_loss(img_embeds, txt_embeds, gamma=self.chamfer_parmeters["mmd_gamma"])
        # diversity loss
        div_loss = batchwise_uniformity_loss(img_embeds, num_embeds=self.chamfer_parmeters["set_size"]) + batchwise_uniformity_loss(txt_embeds, num_embeds=self.chamfer_parmeters["set_size"])

        loss += (tokens_i2t_loss + tokens_t2i_loss) / 2 * self.hash_parameters["triplet_alpha"] + self.chamfer_parmeters["mmd_alpha"] * mmd_loss + self.chamfer_parmeters["unif_alpha"] * div_loss

        
        hash_i2t_loss, hash_t2i_loss, quan_image_loss, quan_text_loss = self.trip_loss(img_hash, txt_hash, labels, indexs, **kwags)
        hash_trip_loss = (hash_i2t_loss + hash_t2i_loss) / 2
        quan_loss = (quan_image_loss + quan_text_loss) / 2

        loss += hash_trip_loss * self.hash_parameters["hash_triplet_alpha"] + quan_loss * self.hash_parameters["quan_alpha"]
        loss_dict = {
            "All loss": loss.data,
            "Tokens": {
                "Similarity": {
                    "i2t": tokens_i2t_loss.data,
                    "t2i": tokens_t2i_loss.data
                    },
                "Maximum Mean Discrepancy": mmd_loss.data,
                "Diversity": div_loss.data if isinstance(div_loss, torch.Tensor) else div_loss
                },
            "Hash": {
                "Triplet": {
                    "i2t": hash_i2t_loss.data if hash_i2t_loss is not None else None,
                    "t2i": hash_t2i_loss.data if hash_t2i_loss is not None else None
                    },
                "Quantization": {
                    "image": quan_image_loss.data,
                    "text": quan_text_loss.data
                    }
                }
            }
        
        return loss, loss_dict


    def object_function(self, img_embeds, img_hash, txt_embeds, txt_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_embeds.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.chamfer_similarity_loss(img_embeds=img_embeds, img_hash=img_hash, txt_embeds=txt_embeds, 
                                            txt_hash=txt_hash, labels=labels, indexs=indexs, **kwags)
    
