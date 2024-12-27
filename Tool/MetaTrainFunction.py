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
    distances = torch.norm(B.unsqueeze(1) - A.unsqueeze(0), dim=2)
    predicted_labels_index = torch.argmin(distances, dim=1)
    predicted_labels = torch.tensor([A_label[i] for i in predicted_labels_index]).to(C.device)
    correct_predictions = (predicted_labels == C).sum().item()
    accuracy = correct_predictions / len(C)
    return accuracy


def compute_anchor_points(features, labels, n_classes):
    anchors = []
    for class_idx in range(n_classes):
        class_features = features[labels == class_idx]
        anchor = class_features.mean(dim=0)
        anchors.append(anchor)
    return torch.stack(anchors)


def MAMl_train_with_Learnable_triplet(
        model, support_images, support_labels, query_images, query_labels,
        inner_step, optimizer, inner_lr, n_classes, alpha, PK, NK, is_train=True,
):
    meta_loss = []
    meta_acc = []
    lossF_s = ImprovedTripletLoss(alpha=alpha)
    lossF_q = ImprovedTripletLoss(alpha=alpha)

    for support_image, support_label, query_image, query_label in zip(
            support_images, support_labels, query_images, query_labels
    ):
        fast_weights = collections.OrderedDict(model.named_parameters())
        fast_weights = move_fast_weights(fast_weights, next(model.parameters()).device)

        anchors_list = []
        anchor_labels_list = []
        for _ in range(inner_step):
            support_features, margin_bounds = functional_call(model, fast_weights, support_image)
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
            anchors_list.append(Anchor_Group)
            anchor_labels_list.append(Anchor_Group_label)
        query_features, margin_bounds = functional_call(model, fast_weights, query_image)
        Anchor_Group_q, Positive_Group, Negative_Group, Anchor_Group_label_q = TraditionalBuildTriadGroup(
            query_features,
            query_label,
            PK=PK,
            NK=NK)
        query_loss = lossF_q(anchors_list[-1], Positive_Group, Negative_Group, anchor_labels_list[-1], margin_bounds)
        query_acc = calculate_accuracy(anchors_list[-1], anchor_labels_list[-1], query_features, query_label)
        meta_loss.append(query_loss)
        meta_acc.append(query_acc)

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
    Test_Model = deepcopy(model)
    meta_loss = []
    meta_acc = []
    lossF_S = ImprovedTripletLoss(alpha=alpha)
    lossF_q = ImprovedTripletLoss(alpha=alpha)

    if issolo:
        anchors_s = []
        anchor_labels_s = []
        margin_bounds_s = []

    for support_image, support_label, query_image, query_label in zip(
            support_images, support_labels, query_images, query_labels
    ):
        fast_weights = collections.OrderedDict(Test_Model.named_parameters())
        fast_weights = move_fast_weights(fast_weights, next(Test_Model.parameters()).device)

        anchors_list = []
        anchor_labels_list = []
        margin_bounds_list = []
        for _ in range(inner_step):
            support_features, margin_bounds = functional_call(Test_Model, fast_weights, support_image)
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
            anchors_list.append(Anchor_Group)
            anchor_labels_list.append(Anchor_Group_label)
            margin_bounds_list.append(margin_bounds)

        if issolo:
            anchors_s.append(anchors_list[-1])
            anchor_labels_s.append(anchor_labels_list[-1])
            margin_bounds_s.append(margin_bounds_list)

        with torch.no_grad():
            query_features, margin_bounds = functional_call(Test_Model, fast_weights, query_image)
            Anchor_Group_q, Positive_Group, Negative_Group, Anchor_Group_label_q = TraditionalBuildTriadGroup(
                query_features,
                query_label,
                PK=PK,
                NK=NK)
            query_loss = lossF_q(anchors_list[-1], Positive_Group, Negative_Group, anchor_labels_list[-1],
                                 margin_bounds)
            query_acc = calculate_accuracy(anchors_list[-1], anchor_labels_list[-1], query_features, query_label)
            meta_loss.append(query_loss)
            meta_acc.append(query_acc)
    meta_acc_list = meta_acc
    meta_loss = torch.stack(meta_loss).mean()

    meta_acc_m = torch.tensor(meta_acc).mean()

    if issolo:
        Test_Model.load_state_dict(fast_weights, strict=False)
        return Test_Model, anchors_s[-1], anchor_labels_s[-1], meta_loss, meta_acc_m, margin_bounds_s[-1]

    return meta_loss, meta_acc_m, meta_acc_list
