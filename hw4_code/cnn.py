import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # Linear layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
