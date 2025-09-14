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

# @torch.no_grad()
def softmax_hash(embed, return_vector=True):
    # assert len(embed.shape) == 2, "the size of input feature must equal to 2"

    if len(embed.shape) == 2:
        embed = embed.view(embed.shape[0], -1, 2)
    else:
        assert embed.shape[-1] == 2, f"softmax hash must input a shape of B,K,2m. It is {embed.shape}"

    embed = embed.view(embed.shape[0], -1, 2)
    hash = torch.softmax(embed, dim=-1).view(embed.shape[0], -1) if return_vector else torch.softmax(embed, dim=-1)
    return hash

# @torch.no_grad()
def tanh_hash(embed):
    return torch.tanh(embed)

LINEAR_SUBSPACE_HASH = {1: {}, 2: {}, 4: {}, 8: {}, 16: {}}
for code_length in LINEAR_SUBSPACE_HASH.keys():
    tmp_hash = {}
    for i in range(2 ** code_length):
        binary_str = bin(i)[2:]
        binary_str = binary_str.zfill(int(code_length))
        binary_list = [1 if int(bit) > 0.5 else -1 for bit in binary_str]
        tmp_hash.update({i: torch.tensor(binary_list, dtype=torch.float)})
    LINEAR_SUBSPACE_HASH.update({code_length: tmp_hash})

import math
# @torch.no_grad()
def linear_subspace_hash(logits):

    batch_size, token_length, length = logits.shape
    hash_length = int(math.log2(length))
    assert hash_length in LINEAR_SUBSPACE_HASH, f"'linear_subspace_hash' function only supports the hash code size in {list(LINEAR_SUBSPACE_HASH.keys())}, but the input hash code length is {hash_length}."

    code_templete = LINEAR_SUBSPACE_HASH[hash_length]

    probability = F.softmax(logits, dim=-1)
    predict_dec = torch.argmax(probability, dim=-1)

    predict_binary = torch.zeros(batch_size, token_length, hash_length)

    for b in range(batch_size):

        hash_key = predict_dec[b, :].tolist()
        # print(hash_key)
        for t, key in enumerate(hash_key):
            # print(self.hash_dict[key].shape)
            predict_binary[b, t, :] = code_templete[key]
    return predict_binary.view(batch_size, -1).to(logits.device)

    
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
    

if __name__ == "__main__":

    logits = torch.randn(2, 4, 16)

    result = linear_subspace_hash(logits=logits)
    print("hash code: \n", result)
    print("-------" * 20)
    print("logits:\n", logits)
