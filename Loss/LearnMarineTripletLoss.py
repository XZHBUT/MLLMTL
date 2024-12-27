import torch
from torch import nn
import torch.nn.functional as F

from Tool.BuildTriadGroup import TraditionalBuildTriadGroup
from Tool.Dataprocess import read_OrginCWRU


# 手动实现欧氏距离矩阵
def euclidean_distance(x1, x2):
    return torch.sqrt(((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2).sum(dim=2) + 1e-8)

class ImprovedTripletLoss(nn.Module):
    def __init__(self, alpha):
        super(ImprovedTripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label, margin_bounds):
        total_loss = 0.0
        for index, i_class in enumerate(Anchor_Group_label):
            pos = positive_groups[index]  # (B, 256)
            neg = negative_groups[index]  # (B, 256)
            anchor = anchor_group[index].unsqueeze(0)  # (1, 256)

            # 计算到正样本和负样本的距离
            d_p = F.pairwise_distance(anchor.expand_as(pos), pos, p=2)
            d_n = F.pairwise_distance(anchor.expand_as(neg), neg, p=2)

            # 计算损失
            margin_i = margin_bounds[i_class]
            term_pos = d_p / (1 + torch.exp(-(d_p - margin_i)))
            term_neg = torch.clamp(margin_i - d_n, min=0) / (1 + torch.exp(-(margin_i - d_n)))

            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            total_loss += loss_i

        return total_loss / len(Anchor_Group_label)  # 平均损失


class ImprovedTripletLoss_2(nn.Module):
    def __init__(self, alpha):
        super(ImprovedTripletLoss_2, self).__init__()
        self.alpha = alpha

    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label, margin_bounds):

        sample_total_loss = 0.0
        for index, i_class in enumerate(Anchor_Group_label):
            pos = positive_groups[index]  # (B, 256)
            neg = negative_groups[index]  # (B, 256)
            anchor = anchor_group[index].unsqueeze(0)  # (1, 256)
            # 计算到正样本和负样本的距离
            d_p = F.pairwise_distance(anchor.expand_as(pos), pos, p=2)
            d_n = F.pairwise_distance(anchor.expand_as(neg), neg, p=2)
            # 计算损失
            margin_i = margin_bounds[i_class]
            term_pos = d_p / (1 + torch.exp(-(d_p - margin_i)))
            term_neg = torch.clamp(margin_i - d_n, min=0) / (1 + torch.exp(-(margin_i - d_n)))
            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            sample_total_loss += loss_i

        sample_total_loss = sample_total_loss / len(Anchor_Group_label)


        # 计算锚点之间的两两欧氏距离矩阵
        pairwise_distances = euclidean_distance(anchor_group, anchor_group)  # 形状 [N, N]
        # 获取类别标签对应的边界值矩阵
        margin_sum_matrix = margin_bounds[Anchor_Group_label].unsqueeze(1) + margin_bounds[Anchor_Group_label].unsqueeze(0)
        # 上三角
        mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1)
        # 筛选有效的距离和对应的 margin_sum
        valid_distances = pairwise_distances[mask == 1]
        valid_margins = margin_sum_matrix[mask == 1]
        # 计算损失项
        term_neg = torch.clamp(valid_margins - valid_distances, min=0) / (
                    1 + torch.exp(-(valid_margins - valid_distances)))
        anchor_dist_loss = term_neg.mean()

        total_loss = (anchor_dist_loss / len(Anchor_Group_label))  + sample_total_loss

        return total_loss  # 平均损失


if __name__ == '__main__':
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
                                                                        SampleLength=1024,
                                                                        SampleNum=40,
                                                                        normal=False,
                                                                        Rate=[0.5, 0.3, 0.2]
                                                                        )

    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)

    Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label = TraditionalBuildTriadGroup(Train_X, Train_Y,
                                                                                                  PK=1, NK=4)
    loss = ImprovedTripletLoss_2(alpha=0.5)

    margin_bounds = torch.tensor([8.0, 9.0, 3.0, 11.0, 12.0, 33.0, 14.0, 2.0, 16.0, 17.0])
    loss(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label,margin_bounds )
