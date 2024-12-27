import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

  
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32)  # Conv (B, 16, L)
        self.bn1 = nn.BatchNorm1d(16)  # BatchNorm
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 16, L/2)

  
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Conv (B, 32, L/2)
        self.bn2 = nn.BatchNorm1d(32)  # BatchNorm
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 32, L/4)


        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)  # Conv (B, 64, L/4)
        self.bn3 = nn.BatchNorm1d(64)  # BatchNorm
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 64, L/8)


        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)  # Conv (B, 64, L/8)
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # MaxPool (B, 64, L/16)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 将 (B, 64, L/16) 转为 (B, 64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.conv1(x)
        x = self.pool1(x)


        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.conv2(x)
        x = self.pool2(x)


        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.conv3(x)
        x = self.pool3(x)


        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.conv4(x)
        x = self.pool4(x)

        x = self.global_pool(x).squeeze(-1)
        x = x.view(x.size(0), -1)

        return x



class CNNModel_DomainSpecificClassifier(nn.Module):
    def __init__(self, num_classes, drouput):
        super(CNNModel_DomainSpecificClassifier, self).__init__()
        self.fc1 = nn.Linear(64, 64)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (B, 32, 256) to (B, 32*256)

        x = self.fc1(x)

        return x





class CNNModel_Mate(nn.Module):
    def __init__(self, num_classes=10, drouput=0.5, initial_margin=8.0):
        super(CNNModel_Mate, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = CNNModel_DomainSpecificClassifier(num_classes=num_classes, drouput=drouput)


    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return  output



