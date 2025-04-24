import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from common.register import registry
from common.calc_utils import calc_label_sim, cosine_similarity, euclidean_similarity
from ..common import Hash, tanh_hash


def weights_init_kaiming(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class HashLayer(Hash):

    def __init__(self, inputDim=2048, outputDim=64, dropout=0.3):

        super(HashLayer, self).__init__(hash_func=tanh_hash)

        self.img_hash = nn.Sequential(
            nn.Linear(inputDim, inputDim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(inputDim // 2, outputDim),
        )

        self.txt_hash = nn.Sequential(
            nn.Linear(inputDim, inputDim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(inputDim // 2, outputDim),
        )
        self.img_hash.apply(weights_init_kaiming)
        self.txt_hash.apply(weights_init_kaiming)

    def encode_img(self, img):

        hash_feature = self.img_hash(img)
        img_hash = super().forward(hash_feature)
        return img_hash
    
    def quantization(self, code):
        return tanh_hash(code)

    def encode_txt(self, txt):

        hash_feature = self.img_hash(txt)
        txt_hash = super().forward(hash_feature)
        return txt_hash
    
    def forward(self, img, txt):

        img_hash = self.encode_img(img)
        txt_hash = self.encode_txt(txt)

        return img_hash, txt_hash



@registry.register_model("Baseline")
class Baseline(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/DCMHT/base.yaml"}

    def __init__(self, 
                cfg,
                outputDim: int=16, 
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                quan_alpha: float = 0.001):
        super().__init__()
        # assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        state_dict, self.backbone = self.load_backbone(clipPath=clipPath, return_patches=False)
        embed_dim = state_dict["text_projection"].shape[1]
        self.hash = HashLayer(inputDim=embed_dim, outputDim=outputDim)
        self.output_dim = outputDim
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
        quan_alpha = cfg.get("quan_alpha", 0.001)

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            quan_alpha=quan_alpha,
        )
        return model
    
    def tanh_hash_loss(self, code):
        
        hash = torch.sign(code.detach())
        return F.mse_loss(code, hash)
    
    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor, indexs=None, **kwags):

        assert (labels is not None) or (label_sim is not None), "parameters of 'labels' and 'label_sim' must be provided each one."
        label_sim = calc_label_sim(labels, labels)
        label_sim = label_sim.to(a.device)
        s_ab = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss_ab = -torch.mean(label_sim * s_ab - torch.log(1 + torch.exp(s_ab)))
        s_ba = 0.5 * torch.matmul(b, a.t()).clamp(min=-64, max=64)
        b_loss_ba = -torch.mean(label_sim * s_ba - torch.log(1 + torch.exp(s_ba)))

        quan_image_loss = self.tanh_hash_loss(a)
        quan_text_loss = self.tanh_hash_loss(b)

        loss = (b_loss_ab + b_loss_ba) / 2 + self.quan_alpha * (quan_text_loss + quan_image_loss) / 2

        loss_dict = {
            "All loss": loss.data,
            "Bayesian": {
                "i2t": b_loss_ab.data,
                "t2i": b_loss_ba.data,
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
        return self.bayesian_loss(img_hash, txt_hash, labels, indexs, **kwags)
    