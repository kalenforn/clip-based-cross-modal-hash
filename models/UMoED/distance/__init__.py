import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes that x and y are l2 normalized"""
    return x.mm(y.t())

class MPdistance(nn.Module):
    def __init__(self, avg_pool):
        super(MPdistance, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1)).cuda(), nn.Parameter(torch.zeros(1)).cuda()
        
    def forward(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        avg_distance = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_distance

def pairwise_distance(img_embs, txt_embs, extreme=False, T=0.01, return_sim=False, mode="cosine"):
    assert len(img_embs.shape) == 3 and len(txt_embs.shape) == 3, "'pairwise_distance' only supports the 3d embedding's computation."

    if extreme:
        img_embs = torch.softmax(img_embs / T, dim=-1)
        txt_embs = torch.softmax(txt_embs / T, dim=-1)
    if mode == "cosine":
        sim = torch.einsum("btl,ktl->btk", [img_embs, txt_embs]).clamp(min=0)
        distance = sim.mean(dim=1) if return_sim else (1 - sim).mean(dim=1)
    else:
        assert not return_sim, "the current method for computing euclidean distance doesn't support return similarity!"
        batch_size, tokens, l = img_embs.shape
        # (batch_size * tokens) x (batch_size * tokens)
        distance_block = torch.cdist(img_embs.view(-1, l), txt_embs.view(-1, l), p=2.0)
        # batch_size x batch_size x tokens x tokens
        distance_block = distance_block.view(batch_size, tokens, batch_size, tokens).permute(0, 2, 1, 3)
        # tokens x tokens
        mask = torch.diag(torch.ones(tokens)).to(distance_block.device)
        # batch_size x batch_size
        distance = (distance_block * mask).mean(dim=-1).mean(dim=-1)
    return distance
    
class SetwiseDistance(nn.Module):
    def __init__(self, img_set_size=64, txt_set_size=64, denominator=2.0, temperature=16, temperature_txt_scale=1, mode="chamfer"):
        super(SetwiseDistance, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator
        self.temperature = temperature
        self.temperature_txt_scale = temperature_txt_scale # used when computing i2t distance
        self.mode = mode
        
        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.mp_dist = MPdistance(self.xy_avg_pool)
        
    def smooth_chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)

        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        # print(self.x_axis_sum_pool(torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))))
        # print(self.x_axis_sum_pool(torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))).shape)
        
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        # print(right_term)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def max_distance_cosine(self, img_embs, txt_embs, **kwargs):
        dist = cosine_sim(img_embs, txt_embs)
        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_distance

    def smooth_chamfer_distance(self, img_embs, txt_embs, **kwargs):
        return self.smooth_chamfer_distance_cosine(img_embs, txt_embs)
    
    def chamfer_distance(self, img_embs, txt_embs, **kwargs):
        return self.chamfer_distance_cosine(img_embs, txt_embs)
    
    def max_distance(self, img_embs, txt_embs, **kwargs):
        return self.max_distance_cosine(img_embs, txt_embs)
    
    def avg_distance(self, img_embs, txt_embs, **kwargs):
        return self.mp_dist(img_embs, txt_embs)
    
    def compute(self, img_embs, txt_embs, **kwargs):

        if "smooth" in self.mode:
            func = self.smooth_chamfer_distance
        elif "chamfer" in self.mode:
            func = self.chamfer_distance
        elif "max" in self.mode:
            func = self.max_distance
        elif "avg" in self.mode:
            func = self.avg_distance
        # this is the only compute the distance function, other is compute the similarity
        elif "pairwise" in self.mode:
            func = pairwise_distance

        return func(img_embs=img_embs, txt_embs=txt_embs, **kwargs)
    
if __name__ == "__main__":

    dist = SetwiseDistance(4, 4)

    a = torch.tensor(
        [[[ 1., 0.2,  2.,  0.2518],
         [ -1.,  10,  3,  2],
         [-53,  22,  -90,  1.],
         [ 3, -7,  2,  1]],

        [[ 1., 0.2,  2.,  0.2518],
         [ -1.,  10,  3,  2],
         [-53,  22,  -90,  1.],
         [ 3, -7,  2,  1]],

        [[ 1., 0.2,  2.,  0.2518],
         [ -1.,  10,  3,  2],
         [-53,  22,  -90,  1.],
         [ 3, -7,  2,  1]]])
    
    a = F.normalize(a, dim=-1)

    # the tokens in a, b are same, but not the point wise similarity.
    b = torch.tensor(
        [[[ 1., 0.2,  2.,  0.2518],
         [-53,  22,  -90,  1.],
         [ -1.,  10,  3,  2],
         [ 3, -7,  2,  1]],

        [[ 1., 0.2,  2.,  0.2518],
         [ -1.,  10,  3,  2],
         [ 3, -7,  2,  1],
         [-53,  22,  -90,  1.]],

        [[ 1., 0.2,  2.,  0.2518],
         [ 3, -7,  2,  1],
         [-53,  22,  -90,  1.],
         [ -1.,  10,  3,  2]]])

    # b = torch.tensor(
    #     [[[ 1., 0.2,  2.,  0.2518],
    #      [-3,  22,  -90,  1.],
    #      [ -1.,  10,  3,  2],
    #      [ 3, -7,  2,  1]],

    #     [[ 1., 0.2,  2.,  0.2518],
    #      [ -1.,  10,  3,  2],
    #      [ 3, -70,  2,  1],
    #      [-53,  22,  -90,  1.]],

    #     [[ 10., 0.2,  2.,  0.2518],
    #      [ 3, -7,  2,  1],
    #      [-53,  22,  -90,  1.],
    #      [ -1.,  10,  3,  2]]])
    b = F.normalize(b, dim=-1)
    
    print(dist.smooth_chamfer_distance(a.view(-1, a.shape[-1]), b.view(-1, b.shape[-1])))
