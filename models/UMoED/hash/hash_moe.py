import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import Hash, softmax_hash, tanh_hash, linear_subspace_hash, weights_init_kaiming
from .block import SoftMoE, SoftMoEDecoderLayer, SoftMoEDecoder

class MeanHashing(nn.Module):

    def __init__(self, kernel_size=8, padding=0, stride=1):
        super().__init__()
        self.pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):

        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return x

def concatenate(hash):
    return hash.view(hash.shape[0], -1)

class TokenHash(Hash):

    def __init__(self, inputTokenDim=32, outputDim=64, embedDim=512, setDim=64, dropout=0.3, decoder_heads=8, decoder_layers=6, 
                 hash_func=None, merge_func=None, expert_layers=3, num_experts=8, slots_per_expert=8, MoE=False, hidden_dim=512):
        super(TokenHash, self).__init__(hash_func=hash_func, merge_func=merge_func)

        if hidden_dim is None or isinstance(hidden_dim, str):
            hidden_dim = embedDim

        if MoE:
            print("using moe")
            decoder_layer = SoftMoEDecoderLayer(d_model=embedDim, nhead=decoder_heads, dropout=dropout, batch_first=True, num_experts=num_experts, slots_per_expert=slots_per_expert)
            self.decoder = SoftMoEDecoder(decoder_layer=decoder_layer, num_layers=decoder_layers)
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=decoder_heads, dropout=dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=decoder_layers)

        if hidden_dim != embedDim:
            self.first_layer = nn.Linear(in_features=embedDim, out_features=hidden_dim)
        else:
            self.first_layer = None
        

        self.decoder_learned_parameters = nn.Parameter(torch.randn(setDim, hidden_dim), requires_grad=True)
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=outputDim)
        #     self.last_layer.apply(weights_init_kaiming)
        # self.token_layer.apply(weights_init_kaiming)
        # self.hash_layer.apply(weights_init_kaiming)
    
    def forward(self, embeds):
        
        ###
        # B,T,L -> B,M,L
        embeds = embeds if self.first_layer is None else F.relu(self.first_layer(embeds))
        decoder_input_embeds = self.decoder_learned_parameters.unsqueeze(0).repeat(embeds.shape[0], 1, 1)
        # print(embeds.shape)
        # embeds = self.dimension_transfor(embeds)
        # print(embeds.shape)
        # embeds = self.token_layer(embeds)
        # embeds = F.relu(embeds)
        # B,M,L -> B,M,K
        embeds = self.decoder(tgt=decoder_input_embeds, memory=embeds, tgt_mask=None, tgt_key_padding_mask=None)
        embeds = self.classifier(embeds)
        

        hash = super().forward(embeds)

        return embeds, hash

class HashLayer(nn.Module):

    def __init__(self, 
                visual_tokens=48, 
                txt_tokens=32, 
                img_feature_size=512, 
                txt_feature_size=512, 
                outputDim=64, 
                setDim=64, 
                dropout=0.3, 
                decoder_heads=8, 
                decoder_layers=6,
                num_experts=8, 
                expert_layers=3, 
                slots_per_expert=8, 
                MoE=False,
                fusion=True,
                hidden_dim=512, 
                hash_func_: str="tanh", 
                merge_func_: str="mean"):

        super(HashLayer, self).__init__()
        if hash_func_ == "softmax":
            hash_func = softmax_hash 
        elif hash_func_ == "tanh" :
            hash_func = tanh_hash
        elif hash_func_ == "linear_subspace":
            hash_func = linear_subspace_hash
        else:
            raise ValueError(f"Unsupported hash func for {hash_func_}")

        if merge_func_ == "mean":
            merge_func = MeanHashing(kernel_size=setDim)
        elif merge_func_ == "concatenate":
            # merge_func = concatenate
            merge_func = None
        else:
            raise ValueError(f"Unsupported merge func for {merge_func_}")

        self.hash_func = hash_func
        self.fusion = fusion
        # self.img_token_hash = TokenHash(inputTokenDim=visual_tokens, outputDim=outputDim, embedDim=img_feature_size, setDim=setDim, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
        #                                 dropout=dropout, hash_func=hash_func, merge_func=merge_func, expert_layers=expert_layers, num_experts=num_experts, slots_per_expert=slots_per_expert)
        # self.txt_token_hash = TokenHash(inputTokenDim=txt_tokens, outputDim=outputDim, embedDim=txt_feature_size, setDim=setDim, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
        #                                 dropout=dropout, hash_func=hash_func, merge_func=merge_func, expert_layers=expert_layers, num_experts=num_experts, slots_per_expert=slots_per_expert)
        if self.fusion:
            print("fusion", fusion)
            self.hash_module = TokenHash(inputTokenDim=visual_tokens + txt_tokens, outputDim=outputDim, embedDim=img_feature_size, setDim=setDim, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
                                        dropout=dropout, hash_func=hash_func, merge_func=merge_func, expert_layers=expert_layers, num_experts=num_experts, slots_per_expert=slots_per_expert, MoE=MoE, hidden_dim=hidden_dim)
        else:
            self.img_token_hash = TokenHash(inputTokenDim=visual_tokens + txt_tokens, outputDim=outputDim, embedDim=img_feature_size, setDim=setDim, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
                                        dropout=dropout, hash_func=hash_func, merge_func=merge_func, expert_layers=expert_layers, num_experts=num_experts, slots_per_expert=slots_per_expert, MoE=MoE, hidden_dim=hidden_dim)
            self.txt_token_hash = TokenHash(inputTokenDim=visual_tokens + txt_tokens, outputDim=outputDim, embedDim=img_feature_size, setDim=setDim, decoder_heads=decoder_heads, decoder_layers=decoder_layers, 
                                        dropout=dropout, hash_func=hash_func, merge_func=merge_func, expert_layers=expert_layers, num_experts=num_experts, slots_per_expert=slots_per_expert, MoE=MoE, hidden_dim=hidden_dim)

    
    def encode_img(self, token_embeds):

        if self.fusion:
            hash_module = self.hash_module
        else:
            hash_module = self.img_token_hash
        token_embeds, token_hash = hash_module(token_embeds)
        # token_embeds, token_hash = self.img_token_hash(token_embeds)
        return token_embeds, token_hash
    
    def token_merge(self, embeds):
        return self.merge_func(embeds)

    def quantization(self, code):
        return self.hash_func(code)
    
    def encode_txt(self, token_embeds):

        # token_embeds, token_hash = self.txt_token_hash(token_embeds)
        if self.fusion:
            hash_module = self.hash_module
        else:
            hash_module = self.img_token_hash
        token_embeds, token_hash = hash_module(token_embeds)
        return token_embeds, token_hash
    
    def encode_imgAndtxt(self, img_embeds, txt_embeds):

        embeds = torch.hstack([img_embeds, txt_embeds])
        token_embeds, token_hash = self.hash_module(embeds)
        return token_embeds, token_hash
    
    def forward(self, img_inp_embeds, txt_inp_embeds):

        img_token_embeds, img_token_hash = self.encode_img(img_inp_embeds)
        txt_token_embeds, txt_token_hash = self.encode_txt(txt_inp_embeds)
        # fusion_token_embeds, fusion_token_hash = self.encode_imgAndtxt(img_embeds=img_inp_embeds, txt_embeds=txt_inp_embeds)

        return img_token_embeds, img_token_hash, txt_token_embeds, txt_token_hash, None, None# fusion_token_embeds, fusion_token_hash
