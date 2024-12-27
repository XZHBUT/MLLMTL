import torch
from Tool.Dataprocess import read_OrginCWRU

def TraditionalBuildTriadGroup(Orig_Sample, Orig_labels, PK=None, NK=None):
    B, L = Orig_Sample.shape
    if PK == None:
        PK = 2
    if NK == None:
        NK = PK
    unique_labels = torch.unique(Orig_labels)
    num_classes = unique_labels.size(0)
    unique_labels_list = unique_labels.tolist()
    class_means = torch.zeros((num_classes, L)).to(Orig_Sample.device)
    positive_group = []

    for index, i_class in enumerate(unique_labels_list):
        class_indices = (Orig_labels == i_class)
        class_samples = Orig_Sample[class_indices]
        if class_samples.size(0) > 0:
            class_means[index] = class_samples.mean(dim=0)
    class_labels = torch.arange(num_classes)
    Anchor_Group = class_means
    Anchor_Group_label = unique_labels_list
    negative_group = []
    for index, i_class in enumerate(unique_labels_list):
        i_anchor = Anchor_Group[index]
        class_Noise_indices = (Orig_labels == i_class)
        class_samples = Orig_Sample[class_Noise_indices]
        A_Pnosie_dis = torch.norm(class_samples - i_anchor, dim=1)
        if A_Pnosie_dis.size(0) < PK:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, A_Pnosie_dis.size(0), largest=True).indices
        else:
            AP_nearest_indices = torch.topk(A_Pnosie_dis, PK, largest=True).indices
        Pnosie_k = class_samples[AP_nearest_indices]
        positive_group.append(Pnosie_k)

        class_other_Noise_indices = (Orig_labels != i_class)
        class_other_samples = Orig_Sample[class_other_Noise_indices]
        A_Nnosie_dis = torch.norm(class_other_samples - i_anchor, dim=1)
        if A_Nnosie_dis.size(0) < NK:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, A_Nnosie_dis.size(0), largest=False).indices
        else:
            AN_nearest_indices = torch.topk(A_Nnosie_dis, NK, largest=False).indices
        Nnosie_k = class_other_samples[AN_nearest_indices]
        negative_group.append(Nnosie_k)

    Positive_Group = positive_group
    Negative_Group = negative_group

    return Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label


