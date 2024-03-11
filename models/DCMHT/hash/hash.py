
import torch
import torch.nn as nn

from ...common import Hash, softmax_hash, tanh_hash, weights_init_kaiming

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ModalityHash(Hash):

    def __init__(self, inputDim=2048, outputDim=64, num_heads=8, batch_first=True, layernorm=True, hash_func=None):
        
        super(ModalityHash, self).__init__(hash_func=hash_func)
        self.bit = outputDim
        self.atten = nn.MultiheadAttention(inputDim, num_heads=num_heads, batch_first=batch_first)
        self.norm = LayerNorm(inputDim) if layernorm else nn.BatchNorm1d(inputDim)
        self.fc2 = nn.Linear(inputDim, outputDim * 2)
        self.fc2.apply(weights_init_kaiming)

    def freezen(self):
        for param in self.atten.parameters():
            param.requires_grid = False
        for param in self.fc2.parameters():
            param.requires_grid = False
    
    def quantization(self, code):
        return self.hash_func(code)

    def forward(self, data):
        
        data = data.view(data.shape[0], 1, data.shape[1])
        embed = self.atten(data, data, data, need_weights=False)[0]
        embed = embed.squeeze()
        embed = self.norm(embed)
        embed = self.fc2(embed)
        embed = torch.relu(embed)
        
        softmax_hash = super().forward(embed)

        return softmax_hash

class HashLayer(nn.Module):

    def __init__(self, 
                feature_size=512, 
                outputDim=64, 
                num_heads=8,
                batch_first=True, 
                hash_func_: str="tanh"):

        super(HashLayer, self).__init__()
        hash_func = softmax_hash if hash_func_ == "softmax" else tanh_hash
        
        self.hash_func = hash_func
        self.img_hash = ModalityHash(inputDim=feature_size, outputDim=outputDim, layernorm=False, num_heads=num_heads, batch_first=batch_first, hash_func=hash_func)
        self.txt_hash = ModalityHash(inputDim=feature_size, outputDim=outputDim, layernorm=True, num_heads=num_heads, batch_first=batch_first, hash_func=hash_func)
    
    def encode_img(self, embeds):

        hash = self.img_hash(embeds)
        return hash

    def quantization(self, code):
        return self.hash_func(code)
    
    def encode_txt(self, embeds):

        hash = self.txt_hash(embeds)
        return hash
    
    def forward(self, img_embeds, txt_embeds):

        img_hash = self.encode_img(img_embeds)
        txt_hash = self.encode_txt(txt_embeds)

        return img_hash, txt_hash