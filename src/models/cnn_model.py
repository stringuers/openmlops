"""CNN Model for CIFAR-10 classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """CNN architecture for CIFAR-10."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(CIFAR10CNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)

        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.drop4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
        x = self.drop3(self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))))
        x = x.view(x.size(0), -1)
        x = self.drop4(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_model(num_classes: int = 10, dropout_rate: float = 0.5) -> CIFAR10CNN:
    return CIFAR10CNN(num_classes=num_classes, dropout_rate=dropout_rate)
