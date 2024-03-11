import os
import torch
import logging
import torch.nn as nn

from ...common.hash import Hash, tanh_hash, weights_init_kaiming


class LinearHash(Hash):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__(hash_func=tanh_hash)
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)
        self.fc.apply(weights_init_kaiming)
    
    def forward(self, data):
        result = self.fc(data)
        return super().forward(self.drop_out(result))


class Pre_Layer(nn.Module):
    def __init__(self, inputdim=2048, nb_class=80):
        super(Pre_Layer, self).__init__()
        self.fc = nn.Linear(inputdim, nb_class)
        self.fc.apply(weights_init_kaiming)

    def forward(self, data):
        pre = self.fc(data)
        return pre
    
class HashLayer(nn.Module):

    def __init__(self, 
                inputDim: int=512, 
                outputDim: int=64,
                num_classes: int=80):

        super(HashLayer, self).__init__()
        
        self.image_hash = LinearHash(inputDim=inputDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=inputDim, outputDim=outputDim)

        self.image_pre = Pre_Layer(inputdim=inputDim, nb_class=num_classes)
        self.text_pre = Pre_Layer(inputdim=inputDim, nb_class=num_classes)
    
    def encode_img(self, embeds):

        hash = self.image_hash(embeds)
        pre = self.image_pre(embeds)
        return hash, pre

    def quantization(self, code):
        return tanh_hash(code)
    
    def encode_txt(self, embeds):

        hash = self.text_hash(embeds)
        pre = self.text_pre(embeds)
        return hash, pre
    
    def forward(self, img_embeds, txt_embeds):

        img_hash, img_pre = self.encode_img(img_embeds)
        txt_hash, txt_pre = self.encode_txt(txt_embeds)

        return img_hash, txt_hash, img_pre, txt_pre

