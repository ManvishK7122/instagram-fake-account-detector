import torch
import torch.nn as nn

class CNN(nn.module):

    def __init__(self):

        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 2,
            kernel_size = 3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2,1)

    def forward(self,x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x,1)

        x = self.fc1(x)

        return x

model = SimpleCNN()

x = torch.randn(1,1,4,4)

output = model(x)

print(output)
    
