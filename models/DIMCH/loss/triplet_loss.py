from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)

    distances = torch.clamp(1 - cos_sim, 0)

    return distances



def _get_triplet_mask(s_labels, t_labels):

    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).to(s_labels.device)
    # print(th.device, ideal_list.device)
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z


    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = sim_pos - sim_neg
    mask = i_equal_j * (1 - i_equal_k)


    return mask, weight


class TripletLoss(nn.Module):
    def __init__(self, reduction):
        super(TripletLoss, self).__init__()

        self.reduction = reduction

    def forward(self, source, s_labels, target=None, t_labels=None, distance=None, margin=0, weight_=True):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        if distance is None:
            pairwise_dist = cos_distance(source, target)
        else:
            pairwise_dist = distance


        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin


        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = _get_triplet_mask(s_labels, t_labels)
        mask = mask.to(triplet_loss.device)
        weight = weight.to(triplet_loss.device) if weight_ else 1
        triplet_loss = weight * mask * triplet_loss
        # triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss.clamp(0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss

def _get_anchor_triplet_mask(s_labels, t_labels, sim=None):
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    :param s_labels: tensor with shape (batch_size, label_num)
    :param t_labels: tensor with shape (batch_size, label_num)
    :return: positive mask and negative mask, `Tensor` with shape [batch_size, batch_size]
    """
    if sim is None:
        sim = (s_labels.mm(t_labels.t()) > 0).float()

    return sim, 1 - sim

class TripletHardLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletHardLoss, self).__init__()

        # self.dis_metric = dis_metric
        self.reduction = reduction
        # self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, distance=None, margin=0, sim=None):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels


        if distance is None:
            pairwise_dist = cos_distance(source, target)
        else:
            pairwise_dist = distance

        # First, we need to get a mask for every valid positive (they should have same label)
        # and every valid negative (they should have different labels)
        mask_anchor_positive, mask_anchor_negative = _get_anchor_triplet_mask(s_labels, t_labels, sim=sim)
        if mask_anchor_positive.device != pairwise_dist.device:
            mask_anchor_positive = mask_anchor_positive.to(pairwise_dist.device)
            mask_anchor_negative = mask_anchor_negative.to(pairwise_dist.device)

        # For each anchor, get the hardest positive
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        # hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

        # For each anchor, get the hardest negative
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        # hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        # print(anchor_positive_dist, anchor_positive_dist)
        triplet_loss = torch.clamp(anchor_positive_dist - anchor_positive_dist + margin, 0.0)

        # Get final mean triplet loss
        if self.reduction is 'mean':
            triplet_loss = triplet_loss.mean()
        elif self.reduction is 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss
