import torch
import torch.nn as nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2) #giam 2 lan
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2) #giam 2 lan
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2) #giam 2 lan
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2) #giam 2 lan
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1600, 1024), #3136 / 64 = 49 = 7* 7
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        # b, c, h, w = x.shape

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    # def conv_sequential(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
    #         nn.BatchNorm2d(num_features=16),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(out_channels, out_channels, 3, padding=1),
    #         nn.BatchNorm2d(num_features=32),
    #         nn.LeakyReLU(),
    #         nn.MaxPool2d(2, 2)  # giam 2 lan
    #     )

if __name__ == '__main__':
    model = CNN()
    model.train()
    sample_input = torch.rand(2, 3, 224, 224)
    result = model(sample_input)
    print(result.shape)
    summary(model, (3, 224, 224))
