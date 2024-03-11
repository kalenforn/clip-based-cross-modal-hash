import torch
import torch.nn as nn

from ...common import Hash, tanh_hash

class LinearHash(Hash):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__(hash_func=tanh_hash)
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return super().forward(self.drop_out(result))

class HashLayer(nn.Module):

    def __init__(self, 
                inputDim=512, 
                outputDim=64):

        super(HashLayer, self).__init__()
        
        self.img_hash = LinearHash(inputDim=inputDim, outputDim=outputDim)
        self.txt_hash = LinearHash(inputDim=inputDim, outputDim=outputDim)
    
    def encode_img(self, embeds):

        hash = self.img_hash(embeds)
        return hash

    def quantization(self, code):
        return tanh_hash(code)
    
    def encode_txt(self, embeds):

        hash = self.txt_hash(embeds)
        return hash
    
    def forward(self, img_embeds, txt_embeds):

        img_hash = self.encode_img(img_embeds)
        txt_hash = self.encode_txt(txt_embeds)

        return img_hash, txt_hash
