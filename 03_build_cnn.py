
import torch
import torch.nn as nn

# Step 1: Build a CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution: 1 input channel → 16 Feature Maps
        # Output: 28x28x16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # Convolution: 16 → 32 Feature Maps
        # Output: 14x14x32 after first pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: Changes negative values to zero
        self.relu = nn.ReLU()

        # Max Pooling: Reduces height and width by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: Converts Feature Maps into a 1D vector
        self.flatten = nn.Flatten()

        # Fully Connected: 32*7*7 features → 10 digit scores
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Input: [Batch, 1, 28, 28]

        x = self.pool(self.relu(self.conv1(x)))
        # Output: [Batch, 16, 14, 14]

        x = self.pool(self.relu(self.conv2(x)))
        # Output: [Batch, 32, 7, 7]

        x = self.flatten(x)
        # Output: [Batch, 1568]

        x = self.fc(x)
        # Output: [Batch, 10] — raw classification scores

        return x


# Step 2: Create the model
model = CNN()

# Step 3: Create one sample MNIST-like image
image = torch.randn(1, 1, 28, 28)

# Step 4: Pass image through the model
output = model(image)

# Print important results
print("Model:", model)
print("Input Shape:", image.shape)
print("Output Shape:", output.shape)
