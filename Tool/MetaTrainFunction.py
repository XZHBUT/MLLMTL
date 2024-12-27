import collections
from copy import deepcopy

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call

from Loss.LearnMarineTripletLoss import ImprovedTripletLoss, ImprovedTripletLoss_2
from Tool.BuildTriadGroup import TraditionalBuildTriadGroup


def move_fast_weights(fast_weights, device):
    return collections.OrderedDict(
        (name, param.to(device)) for name, param in fast_weights.items()
    )


def calculate_accuracy(A, A_label, B, C):
    """
    A: (N, 1024) 类别标记向量，N为类别数
    B: (X, 1024) 样本向量，X为样本数
    C: (X) 样本的真实标签

    返回: 准确率 (float)
    """
    # A.unsqueeze(0) -> (1, N, 1024), B.unsqueeze(1) -> (X, 1, 1024)
    # 通过广播机制计算每个样本与每个类别标记向量之间的欧氏距离
    distances = torch.norm(B.unsqueeze(1) - A.unsqueeze(0), dim=2)  # (X, N)

    # 找到每个样本距离最近的类别索引
    predicted_labels_index = torch.argmin(distances, dim=1)  # (X)
    predicted_labels = torch.tensor([A_label[i] for i in predicted_labels_index]).to(C.device)
    # 计算预测标签与真实标签的准确率
    correct_predictions = (predicted_labels == C).sum().item()
    accuracy = correct_predictions / len(C)

    return accuracy


def compute_anchor_points(features, labels, n_classes):
    """
    计算支持集的锚点（类中心）。
    Args:
        features: 支持集的特征向量, shape = [N, D]
        labels: 支持集的标签, shape = [N]
        n_classes: 类别数
    Returns:
        anchors: 每个类别的锚点向量, shape = [n_classes, D]
    """
    anchors = []
    for class_idx in range(n_classes):
        class_features = features[labels == class_idx]  # 当前类别的所有特征
        anchor = class_features.mean(dim=0)  # 类中心
        anchors.append(anchor)
    return torch.stack(anchors)  # 返回 shape = [n_classes, D]


def MAMl_train_with_Learnable_triplet(
        model, support_images, support_labels, query_images, query_labels,
        inner_step, optimizer, inner_lr, n_classes, alpha, PK, NK, is_train=True,
):
    """
    改进后的 MAML，支持集和查询集均使用三元组损失。
    Args:
        model: 模型
        support_images: 支持集图片
        support_labels: 支持集标签
        query_images: 查询集图片
        query_labels: 查询集标签
        inner_step: 内部训练步数
        optimizer: 优化器
        inner_lr: 内部学习率
        n_classes: 类别数
        margin: 三元组损失的 margin
        is_train: 是否训练
    Returns:
        meta_loss: 元损失
        meta_acc: 元准确率
    """
    meta_loss = []
    meta_acc = []
    lossF_s = ImprovedTripletLoss_2(alpha=alpha)
    lossF_q = ImprovedTripletLoss_2(alpha=alpha)

    for support_image, support_label, query_image, query_label in zip(
            support_images, support_labels, query_images, query_labels
    ):
        # 初始化 fast_weights
        fast_weights = collections.OrderedDict(model.named_parameters())
        fast_weights = move_fast_weights(fast_weights, next(model.parameters()).device)

        # 内部更新
        anchors_list = []  # 用于保存所有迭代的 Anchor
        anchor_labels_list = []  # 用于保存所有迭代的 Anchor_Label
        for _ in range(inner_step):
            # 支持集特征
            support_features, margin_bounds = functional_call(model, fast_weights, support_image)  # shape = [N, D]
            support_loss = 0
            Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label = TraditionalBuildTriadGroup(
                support_features,
                support_label,
                PK=PK,
                NK=NK)
            support_loss = lossF_s(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label, margin_bounds)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict(
                (name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
            # s_acc = calculate_accuracy(Anchor_Group, Anchor_Group_label, support_features, support_label)
            # 保存 Anchor 和 Anchor_Label
            anchors_list.append(Anchor_Group)
            anchor_labels_list.append(Anchor_Group_label)
        # 查询集损失 (三元组损失)
        query_features, margin_bounds = functional_call(model, fast_weights, query_image)
        Anchor_Group_q, Positive_Group, Negative_Group, Anchor_Group_label_q = TraditionalBuildTriadGroup(
            query_features,
            query_label,
            PK=PK,
            NK=NK)
        query_loss = lossF_q(anchors_list[-1], Positive_Group, Negative_Group, anchor_labels_list[-1], margin_bounds)
        # Anchor_q, Anchor_Label_q = OutputMeanSample(query_features, query_label)
        # Anchor_q = anchors_list[-1]
        # Anchor_Label_q = anchor_labels_list[-1]
        query_acc = calculate_accuracy(anchors_list[-1], anchor_labels_list[-1], query_features, query_label)
        meta_loss.append(query_loss)
        meta_acc.append(query_acc)

    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc


def MaML_Finetunning_with_Learnable_triplet(
        model, support_images, support_labels, query_images, query_labels, inner_step, inner_lr,
        alpha, PK, NK, issolo=False
):
    """
    使用三元组损失和锚点的 MAML Fine-tuning，支持查询集梯度更新。
    Args:
        model: 模型
        support_images: 支持集图片
        support_labels: 支持集标签
        query_images: 查询集图片
        query_labels: 查询集标签
        inner_step: 内部训练步数
        inner_lr: 内部学习率
        n_classes: 类别数
        issolo: 是否返回更新后的模型
    Returns:
        meta_loss: 元损失
        meta_acc: 元准确率
    """
    Test_Model = deepcopy(model)
    meta_loss = []
    meta_acc = []
    lossF_S = ImprovedTripletLoss_2(alpha=alpha)
    lossF_q = ImprovedTripletLoss_2(alpha=alpha)

    if issolo:
        anchors_s = []
        anchor_labels_s = []
        margin_bounds_s = []


    for support_image, support_label, query_image, query_label in zip(
            support_images, support_labels, query_images, query_labels
    ):
        fast_weights = collections.OrderedDict(Test_Model.named_parameters())
        fast_weights = move_fast_weights(fast_weights, next(Test_Model.parameters()).device)

        # 内部更新 (支持集)
        anchors_list = []  # 用于保存所有迭代的 Anchor
        anchor_labels_list = []  # 用于保存所有迭代的 Anchor_Label
        margin_bounds_list = []
        for _ in range(inner_step):
            # 支持集特征
            support_features, margin_bounds = functional_call(Test_Model, fast_weights, support_image)  # shape = [N, D]

            Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label = TraditionalBuildTriadGroup(
                support_features,
                support_label,
                PK=PK,
                NK=NK)
            support_loss = lossF_S(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label, margin_bounds)

            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict(
                (name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads)
            )

            # s_acc = calculate_accuracy(Anchor_Group, Anchor_Group_label, support_features, support_label)
            # 保存 Anchor 和 Anchor_Label
            anchors_list.append(Anchor_Group)
            anchor_labels_list.append(Anchor_Group_label)
            margin_bounds_list.append(margin_bounds)

        if issolo:
            anchors_s.append(anchors_list[-1])
            anchor_labels_s.append(anchor_labels_list[-1])
            margin_bounds_s.append(margin_bounds_list)

        # 查询集计算损失 (三元组损失)
        with torch.no_grad():
            # 查询集特征
            query_features, margin_bounds = functional_call(Test_Model, fast_weights, query_image)
            Anchor_Group_q, Positive_Group, Negative_Group, Anchor_Group_label_q = TraditionalBuildTriadGroup(
                query_features,
                query_label,
                PK=PK,
                NK=NK)
            query_loss = lossF_q(anchors_list[-1], Positive_Group, Negative_Group, anchor_labels_list[-1],
                                 margin_bounds)

            # Anchor_q, Anchor_Label_q = OutputMeanSample(query_features, query_label)
            # Anchor_q = anchors_list[-1]
            # Anchor_Label_q = anchor_labels_list[-1]
            query_acc = calculate_accuracy(anchors_list[-1], anchor_labels_list[-1], query_features, query_label)
            meta_loss.append(query_loss)
            meta_acc.append(query_acc)
    meta_acc_list = meta_acc
    meta_loss = torch.stack(meta_loss).mean()

    meta_acc_m = torch.tensor(meta_acc).mean()

    # 在 issolo 模式下，返回更新后的模型
    if issolo:
        Test_Model.load_state_dict(fast_weights, strict=False)
        return Test_Model, anchors_s[-1], anchor_labels_s[-1], meta_loss, meta_acc_m, margin_bounds_s[-1]

    return meta_loss, meta_acc_m, meta_acc_list
