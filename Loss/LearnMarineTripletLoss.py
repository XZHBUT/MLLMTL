import torch
from torch import nn
import torch.nn.functional as F

from Tool.BuildTriadGroup import TraditionalBuildTriadGroup
from Tool.Dataprocess import read_OrginCWRU



def euclidean_distance(x1, x2):
    return torch.sqrt(((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2).sum(dim=2) + 1e-8)


class ImprovedTripletLoss(nn.Module):
    def __init__(self, alpha, r):
        super(ImprovedTripletLoss, self).__init__()
        self.alpha = alpha
        self.r = r

    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label, margin_bounds):

        sample_total_loss = 0.0
        for index, i_class in enumerate(Anchor_Group_label):
            pos = positive_groups[index]  # (B, 256)
            neg = negative_groups[index]  # (B, 256)
            anchor = anchor_group[index].unsqueeze(0)  # (1, 256)
   
            d_p = F.pairwise_distance(anchor.expand_as(pos), pos, p=2)
            d_n = F.pairwise_distance(anchor.expand_as(neg), neg, p=2)

            margin_i = margin_bounds[i_class]
            term_pos = d_p / (1 + torch.exp(-(d_p - margin_i)))
            term_neg = torch.clamp(margin_i - d_n, min=0) / (1 + torch.exp(-(margin_i - d_n)))
            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            sample_total_loss += loss_i

        sample_total_loss = sample_total_loss / len(Anchor_Group_label)



        pairwise_distances = euclidean_distance(anchor_group, anchor_group)  # 形状 [N, N]

        margin_sum_matrix = margin_bounds[Anchor_Group_label].unsqueeze(1) + margin_bounds[Anchor_Group_label].unsqueeze(0)

        mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1)

        valid_distances = pairwise_distances[mask == 1]
        valid_margins = margin_sum_matrix[mask == 1]

        term_neg = torch.clamp(valid_margins - valid_distances, min=0) / (
                    1 + torch.exp(-(valid_margins - valid_distances)))
        anchor_dist_loss = term_neg.mean()

        total_loss = self.r * (anchor_dist_loss / len(Anchor_Group_label))  + (1- self.r) * sample_total_loss

        return total_loss  



