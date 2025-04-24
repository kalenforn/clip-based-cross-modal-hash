import os
import torch
import torch.nn as nn

from .hash.hash import HashLayer
from ..base import BaseModel
from common.register import registry
from common.calc_utils import calc_label_sim, cosine_similarity, euclidean_similarity

@registry.register_model("DCMHT")
class DCMHT(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/DCMHT/base.yaml"}

    def __init__(self, 
                cfg,
                outputDim: int=16, 
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                hash_func: str="softmax",
                vartheta: float=0.75,
                threshold: float=0.1,
                quan_alpha: float = 0.001,
                similarity_function: str="euclidean"):
        super().__init__()
        # assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        embed_dim, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=False)
        self.hash = HashLayer(feature_size=embed_dim, outputDim=outputDim, num_heads=8, batch_first=True, hash_func_=hash_func)
        self.output_dim = outputDim
        self.hash_func = hash_func
        self.vartheta = vartheta
        self.threshold = threshold
        self.similarity_function = similarity_function
        self.quan_alpha = quan_alpha

    def encode_image(self, image):

        image_embed = self.backbone.encode_image(image)
        image_embed = self.hash.encode_img(image_embed)

        return image_embed

    def encode_text(self, text):
        text_embed = self.backbone.encode_text(text)
        text_embed = self.hash.encode_txt(text_embed)

        return text_embed
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        hash_func = cfg.get("hash_func", "softmax")
        vartheta = cfg.get("vartheta", 0.75)
        threshold = cfg.get("threshold", 0.1)
        similarity_function = cfg.get("similarity_function", "euclidean")
        quan_alpha = cfg.get("quan_alpha", 0.001)

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            hash_func=hash_func,
            vartheta=vartheta,
            threshold=threshold,
            quan_alpha=quan_alpha,
            similarity_function=similarity_function
        )
        return model
    
    def similarity_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):

        vartheta = self.vartheta
        threshold = self.threshold
        similarity_function = self.similarity_function

        similarity = cosine_similarity(a, b) if similarity_function == "cosine" else euclidean_similarity(a, b)

        if similarity_function == "euclidean":
            positive_similarity = similarity * label_sim
            negative_similarity = similarity * (1 - label_sim)
            max_value = float(self.output_dim * 2 * vartheta) ** 0.5
            negative_similarity = negative_similarity.clip(max=max_value)
            negative_similarity = torch.tensor([max_value]).expand_as(negative_similarity).to(a.device) * (1 - label_sim) - negative_similarity

            positive_loss = torch.pow(positive_similarity, 2).mean()
            negative_loss = torch.pow(negative_similarity, 2).mean()

            return similarity, positive_loss, negative_loss
        
        elif similarity_function == "cosine":
            similarity = similarity.clip(min=threshold).clip(max=1-threshold)
            sim_loss = -label_sim * torch.log(similarity) - (1-label_sim) * torch.log(1-similarity)
            return similarity, torch.mean(sim_loss), torch.mean(sim_loss)
    
    def soft_argmax_hash_loss(self, code):
        if len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)
        
        hash_loss = 1 - torch.pow(2 * code - 1, 2).mean()
        return hash_loss
    
    def our_loss(self, image, text, labels=None, indexs=None, **kwags):
        # return super().object_function(a, b, labels, indexs)
        label_sim = calc_label_sim(labels, labels)
        if not label_sim.is_cuda:
            label_sim = label_sim.to(image.device)
        
        intra_similarity, intra_positive_loss, intra_negative_loss = self.similarity_loss(image, text, label_sim)
        inter_similarity_i, inter_positive_loss_i, inter_negative_loss_i = self.similarity_loss(image, image, label_sim)
        inter_similarity_t, inter_positive_loss_t, inter_negative_loss_t = self.similarity_loss(text, text, label_sim)

        quan_image_loss = self.soft_argmax_hash_loss(image)
        quan_text_loss = self.soft_argmax_hash_loss(text)

        intra_similarity_loss = (intra_positive_loss + intra_negative_loss)
        inter_similarity_loss = inter_positive_loss_t + inter_positive_loss_i + inter_negative_loss_i + inter_negative_loss_t
        similarity_loss = inter_similarity_loss + intra_similarity_loss

        quan_loss = (quan_image_loss + quan_text_loss) / 2

        loss = similarity_loss + self.quan_alpha * quan_loss

        loss_dict = {
            "All loss": loss.data,
            "Intra": {
                "Positive": intra_positive_loss.data,
                "Negative": intra_negative_loss.data,
                },
            "Inter": {
                "Positive": {
                    "i2t": inter_positive_loss_i.data,
                    "t2i": inter_positive_loss_t.data 
                    },
                "Negative": {
                    "i2t": inter_negative_loss_i.data,
                    "t2i": inter_negative_loss_t.data 
                    }
                },
            "Quan": {
                "Image": quan_image_loss.data,
                "Text": quan_text_loss.data,
                },
            }

        return loss, loss_dict

    def object_function(self, img_hash, txt_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_hash.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.our_loss(img_hash, txt_hash, labels, indexs, **kwags)
    