import os
import torch
import torch.nn as nn

from .hash.hash import HashLayer
from ..base import BaseModel
from common.register import registry
from .loss import HyP

import math
import xlrd

@registry.register_model("DSPH")
class DSPH(BaseModel):

    DEFAULT_CONFIG_FILE = {"base": "confing/DCMHT/base.yaml"}

    def __init__(self, 
                cfg,
                outputDim: int=16, 
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                numclass: int=80,
                hypseed: int=1,
                alpha: float=0):
        super().__init__()
        assert os.path.isfile(clipPath), f"{clipPath} is not exist!"
        self.cfg = cfg
        state_dict, self.clip = self.load_clip(clipPath=clipPath, return_patches=False)
        embed_dim = state_dict["text_projection"].shape[1]
        self.hash = HashLayer(inputDim=embed_dim, outputDim=outputDim)
        self.output_dim = outputDim
        pwd = os.path.dirname(os.path.realpath(__file__))
        sheet = xlrd.open_workbook(os.path.join(pwd, 'loss/codetable.xlsx')).sheet_by_index(0)
        threshold = sheet.row(outputDim)[math.ceil(math.log(numclass, 2))].value
        self.hyp = HyP(numclass=numclass, hypseed=hypseed, alpha=alpha, threshold=threshold, output_dim=outputDim)

    def encode_image(self, image):

        image_embed = self.clip.encode_image(image)
        image_embed = self.hash.encode_img(image_embed)

        return image_embed

    def encode_text(self, text):
        text_embed = self.clip.encode_text(text)
        text_embed = self.hash.encode_txt(text_embed)

        return text_embed
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000):
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        numclass = cfg.get("numclass", 80)
        hypseed = cfg.get("hypseed", 0)
        alpha = cfg.get("alpha", 0.8)

        model = cls(
            cfg=cfg, 
            outputDim=output_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            numclass=numclass,
            hypseed=hypseed,
            alpha=alpha
        )
        return model
    
    def loss(self, image, text, labels=None, indexs=None, **kwags):

        loss = self.hyp(image, text, labels)

        loss_dict = {
            "All loss": loss.data,
            }

        return loss, loss_dict

    def object_function(self, img_hash, txt_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([img_hash.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.loss(img_hash, txt_hash, labels, indexs, **kwags)
    