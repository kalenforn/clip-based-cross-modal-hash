import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hash.hash_moe import HashLayer
from ..base import BaseModel
from common.register import registry
from ..DIMCH.loss.triplet_loss import TripletLoss
from einops import repeat
from .distance import SetwiseDistance
from common.calc_utils import calc_label_sim

@registry.register_model("UMoED")
class UMoED(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/UMoED/UMoED.yaml"}

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
                decoder_heads=8, 
                decoder_layers=6, 
                extreme=False,
                extreme_T=0.01,
                triplet=True,
                distance_mode="cosine", 
                num_experts=8, 
                expert_layers=3, 
                slots_per_expert=8, 
                MoE=False,
                fusion=True, 
                hidden_dim=512,
                **kwags):
        super().__init__()
        
        self.cfg = cfg
        self.hash_func = hash_func
        # if use_clip:
        embed_dim, visual_token_size, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=True)
        img_feature_size=embed_dim
        txt_feature_size=embed_dim

        self.visual_token_size = visual_token_size
        self.txt_token_size = txt_token_size
        self.cls_alpha = cls_alpha
        
        assert outputDim % setDim == 0, f"'outputDim={outputDim}' must be the integer times of 'setDim={setDim}'"
        vocab_size = 2 ** (outputDim // setDim)

        self.hash = HashLayer(visual_tokens=visual_token_size, txt_tokens=txt_token_size, outputDim=vocab_size, img_feature_size=img_feature_size, txt_feature_size=txt_feature_size, 
                                setDim=setDim, dropout=dropout, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
                                hash_func_=hash_func, merge_func_=merge_func, num_experts=num_experts, expert_layers=expert_layers, slots_per_expert=slots_per_expert, MoE=MoE, hidden_dim=hidden_dim, fusion=fusion)
        self.output_dim = outputDim
        self.triplet_loss = TripletLoss(reduction="mean") if triplet else None
        self.distance = SetwiseDistance(img_set_size=setDim, txt_set_size=setDim, 
                                        denominator=cfg.distance.get("denominator", 2.0), temperature=cfg.distance.get("temperature", 16.0), 
                                        temperature_txt_scale=cfg.distance.get("temperature_txt_scale", 1.0), mode=cfg.distance.get("mode", "chamfer"))
        self.chamfer_parmeters = {
            "set_size": setDim, 
            "unif_alpha": self.cfg.chamfer.get("unif_alpha", 0.01),
            "token_triplet_margin": self.cfg.chamfer.get("token_triplet_margin", 0.2)
        }
        self.hash_parameters = {
            "triplet_alpha": self.cfg.hash_pars.get("triplet_alpha", 1), 
            "quan_alpha": self.cfg.hash_pars.get("quan_alpha", 0.001),
        }
        triplet_margin = self.cfg.hash_pars.get("triplet_margin", 0.3)
        self.triplet_margin = triplet_margin
        self.extreme = extreme
        self.extreme_T = extreme_T
        self.distance_mode = distance_mode
    
    def freezen_backone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

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
    
    def encoder_fusion(self, image, text):
        cls_token, seq_tokens, _ = self.backbone.encode_image(image)
        if seq_tokens.shape[0] != self.visual_token_size and seq_tokens.shape[1] != self.visual_token_size:
            tokens = torch.vstack([cls_token.unsqueeze(dim=0), seq_tokens])
        else:
            tokens = seq_tokens
        img_tokens = tokens.permute(1, 0, 2) if tokens.shape[1] != self.visual_token_size else tokens
        _, tokens, _, _ = self.backbone.encode_text(text)
        txt_tokens = tokens.permute(1, 0, 2) if tokens.shape[1] != self.txt_token_size else tokens
        
        fusion_token_embeds, fusion_token_hash = self.hash.encode_imgAndtxt(img_embeds=img_tokens, txt_embeds=txt_tokens)
        return fusion_token_embeds, fusion_token_hash
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000, txt_token_size=32):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        loss_type = cfg.get("loss_type", "l1")
        triplet_margin = cfg.get("triplet_margin", 0.1)
        setDim = cfg.get("setDim", 64)
        dropout = cfg.get("dropout", 0.3)
        hash_func = cfg.get("hash_func", "softmax")
        merge_func = cfg.get("merge_func", "mean")
        cls_alpha = cfg.get("cls_alpha", 0.7)
        if "softmax" in hash_func:
            output_dim *= 2 
        decoder_heads = cfg.get("decoder_heads", 8)
        decoder_layers = cfg.get("decoder_layers", 6)
        extreme = cfg.get("extreme", True)
        extreme_T = cfg.get("extreme_T", 0.01)
        triplet = cfg.get("triplet", True)
        distance_mode = cfg.get("distance_mode", "cosine")
        num_experts = cfg.get("num_experts", 3)
        expert_layers = cfg.get("expert_layers", 8)
        slots_per_expert = cfg.get("slots_per_expert", 8)
        MoE = cfg.get("MoE", False)
        hidden_dim = cfg.get("hidden_dim", 512)
        fusion = cfg.get("fusion", True)

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
            cls_alpha=cls_alpha, 
            decoder_heads=decoder_heads,
            decoder_layers=decoder_layers,
            extreme=extreme,
            extreme_T=extreme_T,
            triplet=triplet,
            distance_mode=distance_mode,
            num_experts=num_experts,
            expert_layers=expert_layers,
            slots_per_expert=slots_per_expert,
            MoE=MoE,
            hidden_dim=hidden_dim,
            fusion=fusion
        )
        return model
    
    def get_features(self, image, text):
        cls_token, seq_tokens, _ = self.backbone.encode_image(image)
        if seq_tokens.shape[0] != self.visual_token_size and seq_tokens.shape[1] != self.visual_token_size:
            tokens = torch.vstack([cls_token.unsqueeze(dim=0), seq_tokens])
        else:
            tokens = seq_tokens
        # print(self.visual_token_size)
        img_token_features = tokens.permute(1, 0, 2) if tokens.shape[1] != self.visual_token_size else tokens
        # print(tokens.shape)
        # token_embeds, token_hash = self.hash.encode_img(tokens)
        _, tokens, _, _ = self.backbone.encode_text(text)
        txt_token_features = tokens.permute(1, 0, 2) if tokens.shape[1] != self.txt_token_size else tokens

        return img_token_features, txt_token_features
    
    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        img_token_features, txt_token_features = self.get_features(image, text)
        img_embeds, img_hash = self.hash.encode_img(img_token_features)
        txt_embeds, txt_hash = self.hash.encode_txt(txt_token_features)
        # fusion_embeds, fusion_hash = self.hash.encode_imgAndtxt(img_token_features, txt_token_features)
        fusion_embeds, fusion_hash = None, None
        
        if return_loss:
            # return self.object_function(img_hash=img_hash, img_embeds=img_embeds, txt_embeds=txt_embeds, txt_hash=txt_hash, fusion_embeds=fusion_embeds fusion_hash=fusion_hash, labels=labels, indexs=indexs)
            return self.object_function(img_embeds=img_embeds, img_hash=img_hash, txt_embeds=txt_embeds, txt_hash=txt_hash, fusion_embeds=fusion_embeds, fusion_hash=fusion_hash, labels=labels, indexs=indexs)
        
        return img_embeds, img_hash, txt_embeds, txt_hash, fusion_embeds, fusion_hash
    
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
    
    def cross_entropy_loss(self, a: torch.Tensor, b: torch.Tensor):

        # label_sim = calc_label_sim(label, label).to(image.device)

        a_soft_label = torch.argmax(torch.softmax(a, dim=-1), dim=-1)
        b_soft_label = torch.argmax(torch.softmax(b, dim=-1), dim=-1)

        ce_loss_i2t = F.cross_entropy(a.permute(0, 2, 1), b_soft_label.to(a.device))
        ce_loss_t2i = F.cross_entropy(b.permute(0, 2, 1), a_soft_label.to(b.device))

        return ce_loss_i2t, ce_loss_t2i

    def bayesian_loss(self, sim, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = sim.clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss
    
    def similarity_loss(self, img_embeds, img_hash, txt_embeds, txt_hash, fusion_embeds, fusion_hash, labels=None, indexs=None, **kwags):
        
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

        B_i, T_i, L_i = img_embeds.shape
        B_t, T_t, L_t = txt_embeds.shape
        
        img_embeds = img_embeds.view(-1, img_embeds.shape[-1])
        txt_embeds = txt_embeds.view(-1, txt_embeds.shape[-1])
        
        img_embeds = F.normalize(img_embeds, dim=-1)
        txt_embeds = F.normalize(txt_embeds, dim=-1)
        
        loss = 0


        
        if self.triplet_loss is not None:
            if self.distance.mode == "pairwise":
                i2t_embed_distance = self.distance.compute(img_embs=img_embeds.view(B_i, T_i, L_i), txt_embs=txt_embeds.view(B_t, T_t, L_t), extreme=self.extreme, T=self.extreme_T, mode=self.distance_mode)
                t2i_embed_distance = self.distance.compute(img_embs=txt_embeds.view(B_t, T_t, L_t), txt_embs=img_embeds.view(B_i, T_i, L_i), extreme=self.extreme, T=self.extreme_T, mode=self.distance_mode)
                it2i_embed_distance = None
                it2t_embed_distance = None
            else:
                i2t_embed_similarity = self.distance.compute(img_embs=img_embeds, txt_embs=txt_embeds)
                t2i_embed_similarity = self.distance.compute(img_embs=txt_embeds, txt_embs=img_embeds)
                i2t_embed_distance = torch.clamp(1 - i2t_embed_similarity, 0)
                t2i_embed_distance = torch.clamp(1 - t2i_embed_similarity, 0)
                it2i_embed_distance = None
                it2t_embed_distance = None
            tokens_i2t_loss = self.triplet_loss(None, labels, target=None, distance=i2t_embed_distance, margin=self.chamfer_parmeters["token_triplet_margin"])
            tokens_t2i_loss = self.triplet_loss(None, labels, target=None, distance=t2i_embed_distance, margin=self.chamfer_parmeters["token_triplet_margin"])
            tokens_it2t_loss = self.triplet_loss(None, labels, target=None, distance=it2t_embed_distance, margin=self.chamfer_parmeters["token_triplet_margin"]) if it2t_embed_distance is not None else 0
            tokens_it2i_loss = self.triplet_loss(None, labels, target=None, distance=it2i_embed_distance, margin=self.chamfer_parmeters["token_triplet_margin"]) if it2i_embed_distance is not None else 0
        else:
            i2t_embed_sim = self.distance.compute(img_embs=img_embeds.view(B_i, T_i, L_i), txt_embs=txt_embeds.view(B_t, T_t, L_t), extreme=self.extreme, T=self.extreme_T, return_sim=True)
            t2i_embed_sim = self.distance.compute(img_embs=txt_embeds.view(B_t, T_t, L_t), txt_embs=img_embeds.view(B_i, T_i, L_i), extreme=self.extreme, T=self.extreme_T, return_sim=True)
            label_sim = calc_label_sim(labels, labels).to(i2t_embed_sim.device)
            tokens_i2t_loss = self.bayesian_loss(sim=i2t_embed_sim, label_sim=label_sim)
            tokens_t2i_loss = self.bayesian_loss(sim=t2i_embed_sim, label_sim=label_sim)
        
        div_loss_i = batchwise_uniformity_loss(img_embeds.view(B_i, T_i, L_i), num_embeds=self.chamfer_parmeters["set_size"])
        div_loss_t = batchwise_uniformity_loss(txt_embeds.view(B_t, T_t, L_t), num_embeds=self.chamfer_parmeters["set_size"])
        div_loss_it = 0
        div_loss = ( div_loss_i + div_loss_t + div_loss_it) / 3

    
        triplet_loss = (tokens_i2t_loss + tokens_t2i_loss + tokens_it2t_loss + tokens_it2i_loss) / 4

        loss += triplet_loss * self.hash_parameters["triplet_alpha"] + self.chamfer_parmeters["unif_alpha"] * div_loss 

        loss_dict = {
            "All loss": loss.data,
            "Tokens": {
                "Similarity": {
                    "i2t": tokens_i2t_loss.data,
                    "t2i": tokens_t2i_loss.data,
                    "it2i": tokens_it2i_loss,
                    "it2t": tokens_it2t_loss,
                    "All": triplet_loss.data * self.hash_parameters["triplet_alpha"]
                    },
                "Diversity": {
                    "i": div_loss_i.data,
                    "t": div_loss_t.data,
                    "it": div_loss_it,
                    "All": div_loss.data * self.chamfer_parmeters["unif_alpha"]
                }
            }
        }
        
        return loss, loss_dict


    def object_function(self, img_embeds, img_hash, txt_embeds, txt_hash, fusion_embeds, fusion_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_embeds.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.similarity_loss(img_embeds=img_embeds, img_hash=img_hash, txt_embeds=txt_embeds, 
                                            txt_hash=txt_hash, fusion_embeds=fusion_embeds, fusion_hash=fusion_hash, labels=labels, indexs=indexs, **kwags)
    
