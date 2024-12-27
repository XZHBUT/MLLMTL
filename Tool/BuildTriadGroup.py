import torch

from Tool.Dataprocess import read_OrginCWRU


def TraditionalBuildTriadGroup(Orig_Sample, Orig_labels, PK=None, NK=None):
    # 该函数服务于传统三元组损失组
    # 获取样本数和特征数
    B, L = Orig_Sample.shape
    if PK == None:
        PK = 2
    if NK == None:
        NK = PK
    # 使用 torch.unique 获取不重复的标签
    unique_labels = torch.unique(Orig_labels)
    # 获取包含类的个数
    num_classes = unique_labels.size(0)
    # 将 unique_labels 转换为 Python 列表
    unique_labels_list = unique_labels.tolist()
    # 初始化用于存储每个类别的均值
    class_means = torch.zeros((num_classes, L)).to(Orig_Sample.device)
    # 遍历每个类别并计算均值,并且记录不含噪声的正样本数量
    positive_group = []

    for index, i_class in enumerate(unique_labels_list):
        # 创建布尔索引数组，用于选择属于当前类别的样本
        class_indices = (Orig_labels == i_class)
        # 提取属于当前类别的所有样本
        class_samples = Orig_Sample[class_indices]  # (8, 1024)
        # positive_group.append(class_samples)
        # 计算当前类别的均值
        if class_samples.size(0) > 0:
            class_means[index] = class_samples.mean(dim=0)
    # 创建标签数组
    class_labels = torch.arange(num_classes)
    Anchor_Group = class_means
    Anchor_Group_label = unique_labels_list
    negative_group = []
    for index, i_class in enumerate(unique_labels_list):
        i_anchor = Anchor_Group[index]
        # 找正样本
        class_Noise_indices = (Orig_labels == i_class)
        # 提取属于当前类别的所有样本
        class_samples = Orig_Sample[class_Noise_indices]
        # print(class_Noise_samples.shape) (80, 1024) 80=8x10
        # 计算欧式距离
        A_Pnosie_dis = torch.norm(class_samples - i_anchor, dim=1)
        # 获取距离最大的 K 个样本的索引
        if A_Pnosie_dis.size(0) < PK:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, A_Pnosie_dis.size(0), largest=True).indices
        else:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, PK, largest=True).indices
        # 提取对应的 K 个最近样本
        Pnosie_k = class_samples[AP_nearest_indices]
        # print(Pnosie_k.shape) torch.Size([K, 1024])
        # 把噪声正样本和干净正样本合并
        # positive_group[index] = torch.cat((positive_group[index], Pnosie_k), dim=0)
        positive_group.append(Pnosie_k)


        # 找负样本
        # other_sample_mean = torch.cat((Anchor_Group[:index], Anchor_Group[index + 1:]), dim=0)
        # print(other_sample_mean_list) (9, 1024)
        # negative_group.append(other_sample_mean)
        class_other_Noise_indices = (Orig_labels != i_class)
        # 提取不属于当前类别的所有样本
        class_other_samples = Orig_Sample[class_other_Noise_indices]
        # print(class_other_Noise_samples.shape) torch.Size([720, 1024]) 720=800-80
        # 计算欧式距离
        A_Nnosie_dis = torch.norm(class_other_samples - i_anchor, dim=1)
        # 获取距离最小的 K 个样本的索引
        if A_Nnosie_dis.size(0) < NK:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, A_Nnosie_dis.size(0), largest=False).indices
        else:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, NK, largest=False).indices
        # 提取对应的 K 个最近样本
        Nnosie_k = class_other_samples[AN_nearest_indices]
        # print(Nnosie_k.shape) torch.Size([NK, 1024])
        # negative_group[index] = torch.cat((negative_group[index], Nnosie_k), dim=0)
        negative_group.append(Nnosie_k)

    Positive_Group = positive_group # 为列表，每个index为对应类的正样本
    Negative_Group = negative_group

    return Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label


if __name__ == '__main__':
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
                                                                        SampleLength=1024,
                                                                        SampleNum=40,
                                                                        normal=False,
                                                                        Rate=[0.5, 0.3, 0.2]
                                                                        )

    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)

    TraditionalBuildTriadGroup(Train_X, Train_Y, PK=1, NK=4)

