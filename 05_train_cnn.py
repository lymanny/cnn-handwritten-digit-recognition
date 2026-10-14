
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CNN Training:
# Teaches a neural network to recognize image patterns
# by adjusting its weights to reduce prediction errors.

# Step 1: Preprocess MNIST Images

# ToTensor:
# Converts images into tensors and scales pixels to [0, 1].

# Normalize:
# Standardizes pixel values using mean and standard deviation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 2: Load Training Dataset

# MNIST: 60,000 training images of handwritten digits (0–9).
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# DataLoader:
# Divides the dataset into smaller groups called batches.
# Batch Size=64 means training with 64 images per batch.
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Step 3: Build the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d: Extracts image features using convolution filters.
        # 1 input channel → 16 output Feature Maps.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # 16 input Feature Maps → 32 output Feature Maps.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: Changes negative values to zero.
        self.relu = nn.ReLU()

        # MaxPooling: Reduces Feature Map height and width by half.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: Converts Feature Maps into a vector per image.
        self.flatten = nn.Flatten()

        # Linear: Produces 10 scores for digits 0–9.
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # Forward Propagation:
    # Passes images through CNN layers to calculate outputs.
    def forward(self, x):
        # 28x28 → 14x14
        x = self.pool(self.relu(self.conv1(x)))

        # 14x14 → 7x7
        x = self.pool(self.relu(self.conv2(x)))

        # Convert Feature Maps into a vector
        x = self.flatten(x)

        # Return 10 raw classification scores
        return self.fc(x)


# Step 4: Create the CNN Model

# Device: Select GPU if available, otherwise CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

# Step 5: Configure Training

# Loss Function:
# Measures the difference between predictions and correct labels.
criterion = nn.CrossEntropyLoss()

# Adam Optimizer:
# Updates model weights using calculated gradients.
# Learning Rate controls the size of weight updates.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Epoch:
# One complete pass through all 60,000 training images.
epochs = 3

# Step 6: Train the CNN

# train() enables training mode.
model.train()

for epoch in range(epochs):
    total_loss = 0.0

    # Process one batch of 64 images at a time.
    for images, labels in train_loader:

        # Move images and labels to CPU or GPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward Propagation:
        # CNN generates 10 scores for each image.
        outputs = model(images)

        # Calculate Loss:
        # Compare predictions with correct digit labels.
        loss = criterion(outputs, labels)

        # Backpropagation:
        # Calculate gradients showing how weights affect loss.
        loss.backward()

        # Optimizer Step:
        # Adam updates weights to help reduce loss.
        optimizer.step()

        # Accumulate loss weighted by batch size.
        total_loss += loss.item() * images.size(0)

    # Calculate average training loss for this epoch.
    avg_loss = total_loss / len(train_dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Step 7: Save the Trained Model

# state_dict():
# Contains the model's learned weights and biases.
#
# torch.save():
# Saves these parameters for future evaluation or prediction.
torch.save(model.state_dict(), "cnn_mnist.pth")

print("Training completed!")
print("Model saved: cnn_mnist.pth")
