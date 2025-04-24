import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base import BaseModel
from .hash import HashLayer
from common.register import registry

@registry.register_model("MITH")
class MITH(BaseModel):
    def __init__(self,
                cfg,
                outputDim: int=16, 
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                hash_func: str="tanh",
                hyper_tokens_intra: float=1,
                hyper_distill: float=1,
                hyper_info_nce: float=50,
                hyper_cls_inter: float=10,
                hyper_quan: float=8,
                hyper_alpha: float=0.01,
                hyper_lambda: float = 0.99,
                dropout: float=0,
                transformer_layers=2,
                activation="gelu",
                top_k_label=8,
                res_mlp_layers=2):
        super().__init__()
        # assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        embed_dim, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=True)
        self.hash = HashLayer(clip_embed_dim=embed_dim, k_bits=outputDim, dropout=dropout, transformer_layers=transformer_layers, 
                              activation=activation, top_k_label=top_k_label, res_mlp_layers=res_mlp_layers)
        self.output_dim = outputDim
        self.hash_func = hash_func
        self.hyper_tokens_intra = hyper_tokens_intra
        self.hyper_distill = hyper_distill
        self.hyper_info_nce = hyper_info_nce
        self.hyper_cls_inter = hyper_cls_inter
        self.hyper_quan = hyper_quan
        self.hyper_alpha = hyper_alpha
        self.hyper_lambda = hyper_lambda

        self.img_buffer_tokens = torch.randn(train_num, outputDim)# .to(self.rank, non_blocking=True)
        self.img_buffer_cls = torch.randn(train_num, outputDim)# .to(self.rank, non_blocking=True)
        
        self.txt_buffer_tokens = torch.randn(train_num, outputDim)# .to(self.rank, non_blocking=True)
        self.txt_buffer_cls = torch.randn(train_num, outputDim)# .to(self.rank, non_blocking=True)

    def encode_image(self, image):

        cls_token, seq_tokens, _ = self.backbone.encode_image(image)
        res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i = self.hash.encode_img(img_cls=cls_token, img_tokens=seq_tokens)

        return res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i

    def encode_text(self, text, key_padding_mask=None):
        if key_padding_mask.device != text.device:
            key_padding_mask = key_padding_mask.to(text.device)
        txt_eos, txt_tokens, _, new_key_padding_mask = self.backbone.encode_text(text, key_padding_mask=key_padding_mask)
        res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t = self.hash.encode_txt(txt_eos, txt_tokens, new_key_padding_mask)

        return res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t
    
    def forward(self, image, text, key_padding_mask=None, labels=None, indexs=None, return_loss=False):
        res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i = self.encode_image(image)
        res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t = self.encode_text(text, key_padding_mask=key_padding_mask)
        
        if return_loss:
            return self.object_function(res_img_cls=res_img_cls, img_cls_hash=img_cls_hash, tokens_hash_i=tokens_hash_i, trans_tokens_i=trans_tokens_i,
                                         res_txt_cls=res_txt_cls, txt_cls_hash=txt_cls_hash, tokens_hash_t=tokens_hash_t, trans_tokens_t=trans_tokens_t, labels=labels, indexs=indexs)
        
        return res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i, res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t

    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        hash_func = cfg.get("hash_func", "softmax")
        hyper_tokens_intra = cfg.get("hyper_tokens_intra", 1.0)
        hyper_distill = cfg.get("hyper_distill", 1.0)
        hyper_info_nce = cfg.get("hyper_info_nce", 50.0)
        hyper_cls_inter = cfg.get("hyper_cls_inter", 10.0)
        hyper_quan = cfg.get("hyper_quan", 8.0)
        hyper_alpha = cfg.get("hyper_alpha", 0.01)
        hyper_lambda = cfg.get("hyper_lambda", 0.99)
        dropout = cfg.get("drop_out", 0.0)
        transformer_layers = cfg.get("transformer_layers", 2)
        activation = cfg.get("activation", "gelu")
        top_k_label = cfg.get("top_k_label", 8)
        res_mlp_layers = cfg.get("res_mlp_layers", 2)

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            dropout=dropout,
            hash_func=hash_func,
            hyper_tokens_intra=hyper_tokens_intra,
            hyper_distill=hyper_distill,
            hyper_info_nce=hyper_info_nce,
            hyper_cls_inter=hyper_cls_inter,
            hyper_quan=hyper_quan,
            hyper_alpha=hyper_alpha,
            hyper_lambda=hyper_lambda,
            transformer_layers=transformer_layers,
            activation=activation,
            top_k_label=top_k_label,
            res_mlp_layers=res_mlp_layers
        )
        return model
    
    def info_nce_loss(self, out_1, out_2, temperature=0.07):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)
    
    def info_nce_loss_bmm(self, out_1, out_2, temperature=0.07):
        # out1: L,N,D
        # out2: L,N,D
        out_1 = out_1.permute(1, 0, 2)  # NLD
        out_2 = out_2.permute(1, 0, 2)  # NLD
        bz = out_1.size(0)

        sim = torch.bmm(out_1, out_2.permute(0, 2, 1))
        sim /= temperature

        word_num = sim.shape[1]

        sim_1 = rearrange(sim, "b n1 n2 -> (b n1) n2")
        sim_2 = rearrange(sim, "b n1 n2 -> (b n2) n1")

        targets = torch.arange(word_num).type_as(out_1).long().repeat(bz)

        loss_1 = F.cross_entropy(sim_1, targets)
        loss_2 = F.cross_entropy(sim_2, targets)

        return 0.5 * (loss_1 + loss_2)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss
    
    def quantization_loss(self, hash_feature, B):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / self.output_dim
    
    def compute_loss(self, res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i,
                        res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t, labels=None, indexs=None, label_sim=None, **kwags):
        
        assert label_sim is not None, "MITH must provide the label similarity"
        if not label_sim.is_cuda:
            label_sim = label_sim.to(img_cls_hash.device)

        if self.img_buffer_cls.device != img_cls_hash.device:
            self.img_buffer_cls = self.img_buffer_cls.to(img_cls_hash.device, non_blocking=True)
            self.img_buffer_tokens = self.img_buffer_cls.to(img_cls_hash.device, non_blocking=True)
            self.txt_buffer_cls = self.img_buffer_cls.to(img_cls_hash.device, non_blocking=True)
            self.txt_buffer_tokens = self.img_buffer_cls.to(img_cls_hash.device, non_blocking=True)
        self.img_buffer_cls[indexs] = img_cls_hash.detach()
        self.txt_buffer_cls[indexs] = txt_cls_hash.detach()
        self.img_buffer_tokens[indexs] = tokens_hash_i.detach()
        self.txt_buffer_tokens[indexs] = tokens_hash_t.detach()
        B = torch.sign(
                (img_cls_hash.detach() * self.hyper_lambda + tokens_hash_i.detach() * (1 - self.hyper_lambda)) + \
                (txt_cls_hash.detach() * self.hyper_lambda + tokens_hash_t.detach() * (1 - self.hyper_lambda)))
        
        tokens_intra_likelihood_i = self.bayesian_loss(self.img_buffer_tokens, tokens_hash_i, label_sim)
        tokens_intra_likelihood_t = self.bayesian_loss(self.txt_buffer_tokens, tokens_hash_t, label_sim)
        tokens_intra_likelihood = self.hyper_tokens_intra * (tokens_intra_likelihood_i + tokens_intra_likelihood_t)

        cls_inter_likelihood_i2t = self.bayesian_loss(self.img_buffer_cls, txt_cls_hash, label_sim)
        cls_inter_likelihood_t2i = self.bayesian_loss(self.txt_buffer_cls, img_cls_hash, label_sim)
        cls_inter_likelihood = self.hyper_cls_inter * (cls_inter_likelihood_i2t + cls_inter_likelihood_t2i)

        H_i = img_cls_hash * 0.5 + tokens_hash_i * 0.5
        H_t = txt_cls_hash * 0.5 + tokens_hash_t * 0.5
        quan_i = self.quantization_loss(H_i, B)
        quan_t = self.quantization_loss(H_t, B)
        quan = self.hyper_quan * (quan_i + quan_t)

        infoNCE_cls = self.info_nce_loss(res_img_cls, res_txt_cls) 
        infoNCE_tokens = self.info_nce_loss_bmm(trans_tokens_i, trans_tokens_t)
        infoNCE = self.hyper_info_nce * (infoNCE_cls + self.hyper_alpha * infoNCE_tokens)

        item_1 = (F.mse_loss(img_cls_hash.detach(), tokens_hash_i, reduction='sum') + \
                  F.mse_loss(txt_cls_hash.detach(), tokens_hash_t, reduction='sum'))
        # 0.1*gradient back to teacher.
        item_2 = 0.1 * (F.mse_loss(img_cls_hash, tokens_hash_i.detach(), reduction='sum') + \
                        F.mse_loss(txt_cls_hash, tokens_hash_t.detach(), reduction='sum'))
        
        distillation = self.hyper_distill * (item_1 + item_2) / (img_cls_hash.shape[0])

        loss = tokens_intra_likelihood + cls_inter_likelihood + quan + infoNCE + distillation

        loss_dict = {
            "All loss": loss.data,
            "LikeHood": {
                    "intra_tokens": {
                        "image": tokens_intra_likelihood_i.data,
                        "text": tokens_intra_likelihood_t.data
                    },
                    "cls_inter": {
                        "image": cls_inter_likelihood_i2t.data,
                        "text": cls_inter_likelihood_t2i.data
                    }
                },
            "Quantization": {
                "image": quan_i.data,
                "text": quan_t.data
                },
            "InfoNCE": {
                    "cls": infoNCE_cls.data,
                    "tokens": infoNCE_tokens.data
                },
            "Distillation": distillation.data
            }
        return loss, loss_dict
    
    def object_function(self, res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i,
                                         res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t, labels=None, indexs=None, label_sim=None, **kwags):
        if labels is None:
            labels = torch.ones([res_img_cls.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.compute_loss(res_img_cls=res_img_cls, img_cls_hash=img_cls_hash, tokens_hash_i=tokens_hash_i, trans_tokens_i=trans_tokens_i,
                                res_txt_cls=res_txt_cls, txt_cls_hash=txt_cls_hash, tokens_hash_t=tokens_hash_t, trans_tokens_t=trans_tokens_t, 
                                labels=labels, indexs=indexs, label_sim=label_sim, **kwags)
