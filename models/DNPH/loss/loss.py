import torch
import torch.nn as nn
from torch.nn import functional as F

class Loss(torch.nn.Module):
    def __init__(self, num_classes=80, output_dim=16, mrg=1.0):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter((torch.randn(num_classes, output_dim) / 8))
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mrg = mrg

    def forward(self, feature_1, feature_2, predict_1, predict_2, label_1, label_2):
        if label_1.device != feature_1.device:
            label_1 = label_1.to(feature_1.device)
            label_2 = label_2.to(feature_1.device)
        feature_all = torch.cat((feature_1, feature_2), dim=0)
        label_all = torch.cat((label_1, label_2), dim=0)
        proxies = F.normalize(self.proxies, p=2, dim=-1)
        feature_all = F.normalize(feature_all, p=2, dim=-1)

        D_ = torch.cdist(feature_all, proxies) ** 2

        mrg = torch.zeros_like(D_)
        mrg[label_all == 1] = mrg[label_all == 1] + self.mrg
        D_ = D_ + mrg

        p_loss = torch.sum(-label_all * F.log_softmax(-D_, 1), -1).mean()

        d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1)) + \
                 self.cross_entropy(predict_2, torch.argmax(label_2, -1))

        loss = p_loss + d_loss
        return loss