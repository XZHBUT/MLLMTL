import torch
import torch.nn as nn
import torch.nn.functional as F

# 特征提取器类
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # 第一层卷积模块
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32)  # Conv (B, 16, L)
        self.bn1 = nn.BatchNorm1d(16)  # BatchNorm
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 16, L/2)

        # 第二层卷积模块
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Conv (B, 32, L/2)
        self.bn2 = nn.BatchNorm1d(32)  # BatchNorm
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 32, L/4)

        # 第三层卷积模块
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)  # Conv (B, 64, L/4)
        self.bn3 = nn.BatchNorm1d(64)  # BatchNorm
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 64, L/8)

        # 第四层卷积模块
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)  # Conv (B, 64, L/8)
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 64, L/16)
        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 将 (B, 64, L/16) 转为 (B, 64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 第一层：Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.conv1(x)
        x = self.pool1(x)

        # 第二层：Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.conv2(x)
        x = self.pool2(x)

        # 第三层：Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.conv3(x)
        x = self.pool3(x)

        # 第四层：Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.conv4(x)
        x = self.pool4(x)
        # 全局平均池化：将 (B, 64, 64) -> (B, 64)
        x = self.global_pool(x).squeeze(-1)
        x = x.view(x.size(0), -1)

        return x

# 主网络，包含特征提取器和线性层
class CNNModel_cls(nn.Module):
    def __init__(self,n_class):
        super(CNNModel_cls, self).__init__()
        self.feature_extractor = FeatureExtractor()
        # self.fc1 = nn.Linear(64, 64)  # Flatten (B, 32, 256) to (B, 32*256), then (B, 256)
        self.fc2 = nn.Linear(64, n_class)      # (B, 256) to (B, 256)

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output shape (B, 256)
        return x

class CNNModel_DomainSpecificClassifier(nn.Module):
    def __init__(self, num_classes, drouput):
        super(CNNModel_DomainSpecificClassifier, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        # self.dropout = nn.Dropout(drouput)
        # self.BN = nn.BatchNorm1d(64)
        # self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (B, 32, 256) to (B, 32*256)
        # x = F.relu(self.BN(self.fc1(x)))
        x = self.fc1(x)
        # x = self.fc2(x)  # Output shape (B, 256)
        return x


class CNNModel_MultiDomainModel(nn.Module):
    def __init__(self, num_classes_1, num_classes_2, num_classes_3, drouput):
        super(CNNModel_MultiDomainModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier_1 = CNNModel_DomainSpecificClassifier(num_classes_1, drouput)
        self.classifier_2 = CNNModel_DomainSpecificClassifier(num_classes_2, drouput)
        self.classifier_3 = CNNModel_DomainSpecificClassifier(num_classes_3, drouput)

    def forward(self, x, domain):
        features = self.feature_extractor(x)
        if domain == 1:
            return features, self.classifier_1(features)
        elif domain == 2:
            return features, self.classifier_2(features)
        elif domain == 3:
            return features, self.classifier_3(features)

class CNNModel(nn.Module):
    def __init__(self, num_classes=10, drouput=0.5, initial_margin=8.0):
        super(CNNModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = CNNModel_DomainSpecificClassifier(num_classes=num_classes, drouput=drouput)
        self.margin_bounds = nn.Parameter(torch.full((num_classes,), initial_margin))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return features, output, self.margin_bounds


class CNNModel_Mate(nn.Module):
    def __init__(self, num_classes=10, drouput=0.5, initial_margin=8.0):
        super(CNNModel_Mate, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = CNNModel_DomainSpecificClassifier(num_classes=num_classes, drouput=drouput)
        self.margin_bounds = nn.Parameter(torch.full((num_classes,), initial_margin))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return  output, self.margin_bounds


# 测试
# model = CNNModel_cls()
# input_signal = torch.randn(8, 1024)  # Batch size B=8, input size 1024
# output = model(input_signal)
#
# print(f"Input shape: {input_signal.shape}")
# print(f"Output shape: {output.shape}")
