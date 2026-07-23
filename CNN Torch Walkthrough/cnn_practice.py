import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# CREATE RANDOM IMAGE
# ---------------------------------------------------

x = torch.randn(1, 1, 28, 28)

print("Original Input Shape:")
print(x.shape)

# ---------------------------------------------------
# CONVOLUTION
# ---------------------------------------------------

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3
)

x = conv(x)

print("After Convolution:")
print(x.shape)

# ---------------------------------------------------
# RELU ACTIVATION
# ---------------------------------------------------

x = F.relu(x)

print("After ReLU:")
print(x.shape)

# ---------------------------------------------------
# MAX POOLING
# ---------------------------------------------------

pool = nn.MaxPool2d(kernel_size=2)

x = pool(x)

print("After Pooling:")
print(x.shape)

# Flattening
x = x.view(x.size(0),-1)
print("After Flattening")
print(x.shape)

fc = nn.Linear(1352,2)

x = fc(x)

print("After Dense Layer")
print(x.shape)

probabilities = F.softmax(x, dim=1)

print(probabilities)