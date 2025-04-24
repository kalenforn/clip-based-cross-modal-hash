import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import Hash, softmax_hash, tanh_hash, weights_init_kaiming

class MeanHashing(nn.Module):

    def __init__(self, kernel_size=8, padding=0, stride=1):
        super().__init__()
        self.pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):

        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return x

class TokenHash(Hash):

    def __init__(self, inputDim=32, outputDim=64, embedDim=512, setDim=64, dropout=0.3, 
                 hash_func=None, merge_func=None, add_global=False):
        super(TokenHash, self).__init__(hash_func=hash_func, merge_func=merge_func)
        self.token_layer = nn.Conv1d(inputDim, setDim, kernel_size=3, stride=1, padding=1)
        self.hash_layer = nn.Sequential(
            nn.Linear(embedDim, embedDim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedDim // 2, outputDim),
        )
        if add_global:
            self.last_layer = nn.Linear(outputDim, outputDim)
            self.last_layer.apply(weights_init_kaiming)
        self.token_layer.apply(weights_init_kaiming)
        self.hash_layer.apply(weights_init_kaiming)
    
    def forward(self, embeds, global_embed=None):
        
        ###
        # B,T,L -> B,M,L
        embeds = self.token_layer(embeds)
        embeds = F.relu(embeds)
        # B,M,L -> B,M,K
        embeds = self.hash_layer(embeds)

        if global_embed is not None:
            global_embed = global_embed.unsqueeze(1).repeat(1, embeds.shape[1], 1)
            embeds += global_embed
            embeds = self.last_layer(F.relu(embeds))

        hash = super().forward(embeds)

        return embeds, hash
    

class HashLayer(nn.Module):

    def __init__(self, 
                visual_tokens=48, 
                txt_tokens=32, 
                feature_size=512, 
                outputDim=64, 
                setDim=64, 
                dropout=0.3, 
                add_global=False, 
                hash_func_: str="tanh", 
                merge_func: str="mean"):

        super(HashLayer, self).__init__()
        hash_func = softmax_hash if hash_func_ == "softmax" else tanh_hash
        merge_func = MeanHashing(kernel_size=setDim)
        
        self.add_global = add_global
        self.hash_func = hash_func
        self.img_token_hash = TokenHash(inputDim=visual_tokens, outputDim=outputDim, embedDim=feature_size, setDim=setDim,
                                        dropout=dropout, hash_func=hash_func, merge_func=merge_func, add_global=add_global)
        self.txt_token_hash = TokenHash(inputDim=txt_tokens, outputDim=outputDim, embedDim=feature_size, setDim=setDim,
                                        dropout=dropout, hash_func=hash_func, merge_func=merge_func, add_global=add_global)
    
    def encode_img(self, token_embeds):

        token_embeds, token_hash = self.img_token_hash(token_embeds)
        return token_embeds, token_hash
    
    def token_merge(self, embeds):
        return self.merge_func(embeds)

    def quantization(self, code):
        return self.hash_func(code)
    
    def encode_txt(self, token_embeds):

        token_embeds, token_hash = self.txt_token_hash(token_embeds)
        return token_embeds, token_hash
    
    def forward(self, img_token_embeds, txt_token_embeds):

        img_token_embeds, img_token_hash = self.encode_img(img_token_embeds)
        txt_token_embeds, txt_token_hash = self.encode_txt(txt_token_embeds)

        return img_token_embeds, img_token_hash, txt_token_embeds, txt_token_hash
