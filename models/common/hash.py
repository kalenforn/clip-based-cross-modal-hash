import torch
import torch.nn as nn
import torch.nn.functional as F

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

def softmax_hash(embed, return_vector=True):
    # assert len(embed.shape) == 2, "the size of input feature must equal to 2"

    if len(embed.shape) == 2:
        embed = embed.view(embed.shape[0], -1, 2)
    else:
        assert embed.shape[-1] == 2, f"softmax hash must input a shape of B,K,2m. It is {embed.shape}"

    embed = embed.view(embed.shape[0], -1, 2)
    hash = torch.softmax(embed, dim=-1).view(embed.shape[0], -1) if return_vector else torch.softmax(embed, dim=-1)
    return hash

def tanh_hash(embed):
    return torch.tanh(embed)
    
class Hash(nn.Module):

    def __init__(self, hash_func=None, merge_func=None):
        """
        merge_func is used for DiHE method.
        """
        super(Hash, self).__init__()
        assert hash_func is not None, "'hash_func': hash function must be provided!"
        self.hash_func = hash_func
        self.merge_func = merge_func

    def forward(self, embeds):
        hash = self.hash_func(embeds) if self.merge_func is None else self.hash_func(self.merge_func(embeds))
        return hash