import os

import numpy as np
import torch
import torch.nn as nn

from .CLIP.model import build_model as build_clip


class BaseModel(nn.Module):

    DEFAULT_CONFIG_FILE = {"base": "confing/base.yaml"}
    
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
    
    def load_backbone(self, clipPath: str, return_patches: bool=False) -> tuple:

        if os.path.exists(clipPath):
            try:
                model = torch.jit.load(clipPath, map_location="cpu").eval()
                state_dict = model.state_dict()
            except RuntimeError:
                state_dict = torch.load(clipPath, map_location="cpu")
            if return_patches:
                return state_dict["text_projection"].shape[1], state_dict["visual.positional_embedding"].shape[0], build_clip(state_dict, return_patches=return_patches)
            return state_dict["text_projection"].shape[1], build_clip(state_dict, return_patches=return_patches)
        else:
            print("pretrained CLIP model doesn't exist!")
            exit(0)
            
    
    # must be rewrite
    def encode_image(self, x):
        raise NotImplementedError()

    # must be rewrite
    def encode_text(self, x):
        raise NotImplementedError()
    
    # must be rewrite
    def object_function(self, a, b, labels=None, indexs=None, **kwags):
        raise NotImplementedError()
    
    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        if return_loss:
            return self.object_function(image_embed, text_embed, labels=labels, indexs=indexs)
        return image_embed, text_embed
    
    @classmethod
    def from_config(cls, cfg, output_dim=None, train_num=None):
        raise NotImplementedError()
    
    def freezen(self):
        for para in self.parameters():
            para.requires_grad = False
    
    def unfreezen(self):
        for para in self.parameters():
            para.requires_grad = True
    
    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.DEFAULT_CONFIG_FILE
        ), "Unknown model type {}".format(model_type)
        return os.path.join(cls.DEFAULT_CONFIG_FILE[model_type])