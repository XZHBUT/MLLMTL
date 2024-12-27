import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns




def plot_confusion_matrix(model, test_data, test_labels, anchors, anchor_labels):
    """
    生成并可视化模型的混淆矩阵。

    model: 已训练的模型
    test_data: 测试数据 (B, 256)
    test_labels: 测试标签 (B,)
    anchors: (10, 256) 每个类的锚点
    anchor_labels: (10,) 每个锚点的标签 (0-9)

    使用 B007, B014, ..., normal 作为类别名
    """

    # 类别名称对应的标签
    class_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'Normal']

    # 确保数据和模型在同一设备上 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    anchors = anchors.to(device)  # 确保锚点在同一设备上

    # 将模型设为评估模式，并禁用梯度计算
    model.eval()
    with torch.no_grad():
        # 获取模型输出
        _, outputs, _ = model(test_data)  # 输出的形状为 (B, 256)

        # 计算测试样本与每个锚点的欧氏距离，使用 torch.cdist
        distances = torch.cdist(outputs, anchors)  # 结果形状 (B, 10)，每一行是到10个锚点的距离

        # 找到每个样本最近的锚点（最小距离的索引）
        closest_anchor_indices = torch.argmin(distances, dim=1)  # 返回最小距离的锚点索引

        # 根据锚点索引得到预测的类标签
        predictions = torch.tensor([anchor_labels[idx] for idx in closest_anchor_indices.cpu()]).to(device)

    # 生成混淆矩阵
    cm = confusion_matrix(test_labels.cpu().numpy(), predictions.cpu().numpy())

    # 将混淆矩阵转换为百分比
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8.5))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # Oranges

    # 放大标题、刻度、标签的字体大小
    plt.xlabel('Predicted Label', fontsize=17)
    plt.ylabel('True Label', fontsize=17)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=45)

    # 显示图形
    plt.show()

def plot_confusion_matrix_hit(y_true, y_pred):
    # class_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']
    class_names = ['OR', 'IR05', 'IR10']
    """
    绘制混淆矩阵，百分比表示，字体大且加粗，对角线用白色，主色调为黄色。
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    """
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_percentage, cmap='Blues')  # 主色调为黄色

    # fig.colorbar(cax)

    # 添加百分比数字
    for (i, j), val in np.ndenumerate(cm_percentage):
        color = "white" if i == j else "black"  # 对角线白色，其余黑色
        ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=20, color=color)

    # 设置轴标签和标题
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16, rotation=90)
    ax.set_xlabel('Predicted', fontsize=20)
    ax.set_ylabel('True', fontsize=20)
    ax.xaxis.set_ticks_position('bottom')  # 将 Predicted 刻度显示在下方
    # 自动调整布局，避免标签超出画布
    plt.tight_layout()

    plt.show()


def plot_confusion_matrix_HUST(y_true, y_pred):
    # class_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']
    # class_names = ['OR', 'IR05', 'IR10']
    class_names = ['Broken Tooth', 'Missing Tooth', 'Healthy']
    """
    绘制混淆矩阵，百分比表示，字体大且加粗，对角线用白色，主色调为黄色。
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    """
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_percentage, cmap='Blues')  # 主色调为黄色

    # fig.colorbar(cax)

    # 添加百分比数字
    for (i, j), val in np.ndenumerate(cm_percentage):
        color = "white" if i == j else "black"  # 对角线白色，其余黑色
        ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=25, color=color)

    # 设置轴标签和标题
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16, rotation=90)
    ax.set_xlabel('Predicted', fontsize=20)
    ax.set_ylabel('True', fontsize=20)
    ax.xaxis.set_ticks_position('bottom')  # 将 Predicted 刻度显示在下方
    # 自动调整布局，避免标签超出画布
    plt.tight_layout()

    plt.show()
