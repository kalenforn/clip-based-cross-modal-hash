import torch
import numpy as np
from typing import Union

from sklearn.metrics.pairwise import euclidean_distances


def calc_label_sim(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    return (a.matmul(b.transpose(0, 1)) > 0).float()

def generate_weight_sim(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    sim_origin = a.matmul(b.transpose(0, 1))
    batch_size = a.shape[0]
    label_sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).to(a.device)
    # print(th.device, ideal_list.device)
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z

    return label_sim, sim_origin

def euclidean_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        similarity = torch.cdist(a, b, p=2.0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        similarity = euclidean_distances(a, b)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    return similarity

def cosine_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / a.norm(dim=-1, keepdim=True) # if len(torch.where(a != 0)[0]) > 0 else a
        b = b / b.norm(dim=-1, keepdim=True) # if len(torch.where(b != 0)[0]) > 0 else b
        return torch.matmul(a, b.t())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) # if len(np.where(a != 0)[0]) > 0 else a
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) # if len(np.where(b != 0)[0]) > 0 else b
        return np.matmul(a, b.T)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    
def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    
    num_query = query_L.shape[0]
    
    if isinstance(qB, torch.Tensor) and qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    if query_L.device != qB.device:
        query_L = query_L.to(qB.device)
        retrieval_L = retrieval_L.to(qB.device)
    
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    gnds = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
    if gnds.device != qB.device:
        gnds = gnds.to(qB.device)
    tsums = torch.sum(gnds, dim=-1, keepdim=True, dtype=torch.int32)
    hamms = calc_hammingDist(qB, rB)
    _, ind = torch.sort(hamms, dim=-1)
    # if ind.device != gnds.device:
    #     totals = ind.to(gnds.device)

    totals = torch.min(tsums, torch.tensor([k], dtype=torch.int32).expand_as(tsums).to(tsums.device))
    # if totals.device != gnds.device:
    #     totals = totals.to(gnds.device)
    for iter in range(num_query):
        gnd = gnds[iter][ind[iter]]
        total = totals[iter].squeeze()
        count = torch.arange(1, total + 1).type(torch.float32).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count / tindex)
    map = map / num_query

    return map
