import os
import torch.nn as nn
import numpy as np
import torch
from ..base import BaseModel

from ..DCMHT.hash import HashLayer

from common.register import registry

@registry.register_model("TwDH")
class TwDH(BaseModel):

    def __init__(self, 
                cfg, 
                long_dim: int=512, 
                short_dim: int=16,
                clipPath: str="./ViT-B-32.pt",
                train_num: int=10000,
                hash_func: str="softmax",
                long_center: str="./data/transformer/TwDH/center/long",
                short_center: str="./data/transformer/TwDH/center/short",
                trans: str="./data/transformer/TwDH/center/trans", 
                quan_alpha: float=0.5,
                low_rate: float=0):

        super(TwDH, self).__init__()

        self.cfg = cfg
        state_dict, self.clip = self.load_clip(clipPath=clipPath, return_patches=False)
        embed_dim = state_dict["text_projection"].shape[1]
        self.hash = HashLayer(feature_size=embed_dim, outputDim=long_dim, num_heads=8, batch_first=True, hash_func_=hash_func)
        self.output_dim = long_dim

        # print(long_center, short_center, trans)
        self.long_center = torch.load(long_center).float()
        # print(self.long_center)
        if os.path.isfile(short_center):
            key = os.path.basename(short_center).strip().split(".")[0]
            self.short_center = {key: torch.load(short_center).float()}
        else:
            self.short_center = {}
            for item in os.listdir(short_center):
                key = item.strip().split(".")[0]
                self.short_center.update({key: torch.load(os.path.join(short_center, item)).float()})
        # print(self.short_center)

        if os.path.isfile(trans):
            key = os.path.basename(trans).strip().split(".")[0]
            self.trans = {key: torch.load(trans).float()}
        else:
            self.trans = {}
            for item in os.listdir(trans):
                key = item.strip().split(".")[0]
                self.trans.update({key: torch.load(os.path.join(trans, item)).float()})
        # print(self.trans)
        self.quan_alpha = quan_alpha
        self.low_rate = low_rate
        self.criterion = nn.BCELoss()
        self.short_dims = [int(k) for k in self.short_center]
        # print(self.short_dims)
        # print(self.short_center)
    
    def get_short_dims(self):
        return self.short_dims
    
    def encode_image(self, image):

        image_embed = self.clip.encode_image(image)
        long_hash = self.hash.encode_img(image_embed)
        short_hash = {}
        for k, v in self.trans.items():
            v = v.to(long_hash.device)
            short_hash.update({k: self.hash.quantization(long_hash.matmul(v))})
        # print(short_hash)
        return long_hash, short_hash

    def encode_text(self, text):
        text_embed = self.clip.encode_text(text)
        long_hash = self.hash.encode_txt(text_embed)
        short_hash = {}
        for k, v in self.trans.items():
            v = v.to(long_hash.device)
            short_hash.update({k: self.hash.quantization(long_hash.matmul(v))})

        return long_hash, short_hash
    
    @classmethod
    def from_config(cls, cfg, output_dim=16, train_num=10000):
        long_dim = cfg.get("long_dim", 512)
        short_dim = output_dim
        clip_path = cfg.get("clip_path", "./ViT-B-32.pt")
        hash_func = cfg.get("hash_func", "softmax")
        long_center = cfg.get("long_center", "./data/transformer/TwDH/center/long")
        short_center = cfg.get("short_center", "./data/transformer/TwDH/center/short")
        trans_matrix = cfg.get("trans_matrix", "./data/transformer/TwDH/center/trans")
        long_center = os.path.join(long_center, str(long_dim) + ".pkl")
        trans_matrix = os.path.join(trans_matrix, str(long_dim))

        quan_alpha = cfg.get("quan_alpha", 0.5)
        low_rate = cfg.get("low_rate", 0)

        return cls(
            cfg=cfg, 
            long_dim=long_dim,
            short_dim=short_dim, 
            clipPath=clip_path, 
            train_num=train_num,
            hash_func=hash_func,
            long_center=long_center,
            short_center=short_center,
            trans=trans_matrix,
            quan_alpha=quan_alpha,
            low_rate=low_rate
        )

    def forward(self, image, text, labels=None, indexs=None, return_loss=False):
        img_long_hash, img_short_hash = self.encode_image(image)
        txt_long_hash, txt_short_hash = self.encode_text(text)
        
        if return_loss:
            return self.object_function(long_img_hash=img_long_hash, long_txt_hash=txt_long_hash, short_img_hash=img_short_hash, short_txt_hash=txt_short_hash, labels=labels, indexs=indexs)
        
        return img_long_hash, img_short_hash, txt_long_hash, txt_short_hash
    
    def soft_argmax_hash_loss(self, code):
        if len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)
        
        hash_loss = 1 - torch.pow(2 * code - 1, 2).mean()
        return hash_loss
    
    def compute_loss(self, long_img_hash, long_txt_hash, short_img_hash, short_txt_hash, labels, indexs, **kwags):
        
        long_hash_label = hash_convert(hash_center_multilables(labels, self.long_center)).float().to(long_img_hash.device, non_blocking=True)
        long_image_loss = self.criterion(long_img_hash, long_hash_label)
        long_text_loss = self.criterion(long_txt_hash, long_hash_label)

        long_nce_loss = (long_image_loss + long_text_loss) / 2

        long_code_image_loss = self.soft_argmax_hash_loss(long_img_hash)
        long_code_text_loss = self.soft_argmax_hash_loss(long_txt_hash)
        long_quan_loss = (long_code_image_loss + long_code_text_loss) / 2

        short_nce_loss = {}
        short_quan_loss = {}
        for k, v in self.short_center.items():
            short_hash_label = hash_convert(hash_center_multilables(labels, v)).float().to(long_img_hash.device, non_blocking=True)

            short_image_loss = self.criterion(short_img_hash[k], short_hash_label)
            # print(short_img_hash[k])
            short_text_loss = self.criterion(short_txt_hash[k], short_hash_label)
            # print(short_txt_hash[k])
            short_nce_loss.update({k: (short_image_loss + short_text_loss) / 2})

            short_code_image_loss = self.soft_argmax_hash_loss(short_img_hash[k])
            short_code_text_loss = self.soft_argmax_hash_loss(short_txt_hash[k])
            short_quan_loss.update({k: (short_code_image_loss + short_code_text_loss) / 2})

        
        loss = long_nce_loss + self.quan_alpha * long_quan_loss
        for k, v in short_nce_loss.items():
            loss += self.low_rate * v
        for k, v in short_quan_loss.items():
            loss += self.low_rate * v
        
        short_dict = {}
        for k, v in short_nce_loss.items():
            short_dict.update({k:{"NCE": v, "Quan": short_quan_loss[k]}})

        loss_dict = {
            "All loss": loss.data,
            "Long": {
                "NCE":{
                    "image": long_image_loss.data,
                    "text": long_text_loss.data,
                    },
                "Quan":{
                    "image": long_code_image_loss.data,
                    "text": long_code_text_loss.data,
                    }
                },
            "Short": short_dict,
            }
        return loss, loss_dict

    def object_function(self, long_img_hash, long_txt_hash, short_img_hash, short_txt_hash, labels=None, indexs=None, **kwags):
        if labels is None:
            labels = torch.ones([long_img_hash.shape[0]], dtype=torch.int)
            labels = labels.diag()
        return self.compute_loss(long_img_hash, long_txt_hash, short_img_hash, short_txt_hash, labels, indexs, **kwags)
    
def hash_center_multilables(labels, Hash_center):
    # if labels.device != Hash_center.device:
    #     Hash_center = Hash_center.to(labels.device)
    is_start = True
    random_center = torch.randint_like(Hash_center[0], 2)
    for label in labels:
        one_labels = (label == 1).nonzero() 
        one_labels = one_labels.squeeze(1)
        Center_mean = torch.mean(Hash_center[one_labels], dim=0)
        Center_mean[Center_mean<0] = -1
        Center_mean[Center_mean>0] = 1
        random_center[random_center==0] = -1   
        Center_mean[Center_mean == 0] = random_center[Center_mean == 0]  
        Center_mean = Center_mean.view(1, -1) 

        if is_start: 
            hash_center = Center_mean
            is_start = False
        else:
            hash_center = torch.cat((hash_center, Center_mean), 0)
    # hash_center = hash_center.to(labels.device)
    # print(hash_center.device)
    return hash_center

def hash_convert(hash_label):
    if len(hash_label.shape) == 2:
        result = torch.zeros([hash_label.shape[0], hash_label.shape[1] ,2])
        hash_label = (hash_label > 0).long()
        i = torch.arange(hash_label.shape[0]).view(hash_label.shape[0], -1).expand_as(hash_label)
        j = torch.arange(hash_label.shape[1]).expand_as(hash_label)
        result[i, j, hash_label] = 1
        result = result.view(hash_label.shape[0], -1)
    elif len(hash_label.shape) == 1:
        result = torch.zeros([hash_label.shape[0], 2])
        hash_label = (hash_label > 0).long()
        result[torch.arange(hash_label.shape[0]), hash_label] = 1
        result = result.view(hash_label.shape[0], -1)
    result = result.to(hash_label.device)
    return result
