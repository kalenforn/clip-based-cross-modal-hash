import os
import torch
import torch.nn as nn

from .hash.hash import HashLayer
from ..base import BaseModel
from common.register import registry
from .loss import Loss, gene_noise, rand_unit_rect

@registry.register_model("DNPH")
class DNPH(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/DCMHT/base.yaml"}

    def __init__(self, 
                cfg,
                outputDim: int=16, 
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                numclass: int=80,
                mrg: int=1.0,
                noise_alpha: float=0.1):
        super().__init__()
        # assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        embed_dim, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=False)
        self.hash = HashLayer(inputDim=embed_dim, outputDim=outputDim)
        self.output_dim = outputDim
        self.loss = Loss(num_classes=numclass, mrg=mrg, output_dim=outputDim)
        self.noise_alpha = noise_alpha

    def encode_image(self, image):

        image_embed = self.backbone.encode_image(image)
        img_hash, img_pre = self.hash.encode_img(image_embed)

        return img_hash, img_pre

    def encode_text(self, text):
        text_embed = self.backbone.encode_text(text)
        txt_hash, txt_pre = self.hash.encode_txt(text_embed)

        return txt_hash, txt_pre
    
    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        img_hash, img_pre = self.encode_image(image)
        txt_hash, txt_pre = self.encode_text(text)
        
        if return_loss:
            return self.object_function(img_hash=img_hash, txt_hash=txt_hash, img_pre=img_pre, txt_pre=txt_pre, labels=labels, indexs=indexs)
        
        return img_hash, txt_hash, img_pre, txt_pre
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        numclass = cfg.get("numclass", 80)
        mrg = cfg.get("mrg", 1.0)
        noise_alpha = cfg.get("noise_alpha", 1.0)

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            numclass=numclass,
            mrg=mrg,
            noise_alpha=noise_alpha
        )
        return model
    
    def compute_loss(self, img_hash, txt_hash, img_pre, txt_pre, labels=None, indexs=None, **kwags):
        # return super().object_function(a, b, labels, indexs)

        loss1 = self.loss(img_hash, txt_hash, img_pre, txt_pre, labels, labels)

        batch_size_, code_length = img_hash.shape
        s_vector = rand_unit_rect(batch_size_, code_length)

        i_noises = gene_noise(img_hash.cpu().detach().numpy(), s_vector)
        t_noises = gene_noise(txt_hash.cpu().detach().numpy(), s_vector)

        i_noises = torch.from_numpy(i_noises).float().to(img_hash.device)
        t_noises = torch.from_numpy(t_noises).float().to(img_hash.device)
        i_noise_loss = img_hash.mul(i_noises).sum(dim=-1).mean()
        t_noise_loss = txt_hash.mul(t_noises).sum(dim=-1).mean()
        noise_loss = i_noise_loss + t_noise_loss

        loss = loss1 - self.noise_alpha * noise_loss

        loss_dict = {
            "All loss": loss.data,
            "Noise": {
                "image": i_noise_loss,
                "text": t_noise_loss
            }
        }

        return loss, loss_dict

    def object_function(self, img_hash, txt_hash, img_pre, txt_pre, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_hash.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.compute_loss(img_hash, txt_hash, img_pre, txt_pre, labels, indexs, **kwags)
    