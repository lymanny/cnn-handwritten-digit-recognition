
import torch
import torch.nn as nn

# Forward Propagation:
# Passes input images through CNN layers to calculate outputs.
# It does NOT update model weights.

# Step 1: Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d: Extracts image features using convolution filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: Changes negative values to zero
        self.relu = nn.ReLU()

        # MaxPool2d: Reduces Feature Map height and width
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: Converts Feature Maps into a 1D vector per image
        self.flatten = nn.Flatten()

        # Linear: Converts 1568 features into 10 digit scores
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # Step 2: Define Forward Propagation
    def forward(self, x):
        # Input: [Batch, 1, 28, 28]
        print("Input:", x.shape)

        # Convolution extracts features
        # ReLU applies activation
        # Pooling reduces the spatial size
        x = self.pool(self.relu(self.conv1(x)))
        print("After Conv1 + ReLU + Pool:", x.shape)

        # Second Convolution block extracts additional features
        x = self.pool(self.relu(self.conv2(x)))
        print("After Conv2 + ReLU + Pool:", x.shape)

        # Flatten converts [Batch, 32, 7, 7] to [Batch, 1568]
        x = self.flatten(x)
        print("After Flatten:", x.shape)

        # Fully Connected Layer generates 10 raw class scores
        x = self.fc(x)
        print("After FC:", x.shape)

        return x


# Step 3: Create the CNN Model
model = CNN()

# Step 4: Create one sample MNIST-like image
# Batch=1, Channels=1, Height=28, Width=28
image = torch.randn(1, 1, 28, 28)

# Step 5: Run Forward Propagation
# no_grad() disables gradient tracking (no training)
with torch.no_grad():
    output = model(image)
