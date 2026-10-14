
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CNN Evaluation:
# Tests a trained CNN on images it did not train on.
# Accuracy = Correct Predictions / Total Images * 100

# Step 1: Preprocess MNIST Test Images
# Use the SAME normalization as during training.
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert pixels from 0-255 to 0-1
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 2: Load MNIST Test Dataset
# train=False selects 10,000 test images.
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# DataLoader:
# Divides test images into batches of 64.
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Step 3: Define the SAME CNN Architecture as Lesson 5
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract 16 Feature Maps from grayscale images.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # Extract 32 Feature Maps from previous features.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: Changes negative values to zero.
        self.relu = nn.ReLU()

        # Max Pooling: Reduces height and width by half.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: Converts Feature Maps into a vector.
        self.flatten = nn.Flatten()

        # Linear: Produces 10 classification scores (digits 0-9).
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # Forward Propagation:
    # Pass images through the CNN to produce digit scores.
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


# Step 4: Load the Trained Model

# Select GPU if available; otherwise use CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

# torch.load(): Reads saved model weights from Lesson 5.
# load_state_dict(): Applies saved weights to the CNN.
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)

# eval(): Activates evaluation mode.
model.eval()

# Step 5: Evaluate the CNN
correct = 0
total = 0

# inference_mode(): Disables gradient tracking.
# Evaluation does NOT train or update weights.
with torch.inference_mode():

    for images, labels in test_loader:
        # Move the batch to CPU or GPU.
        images = images.to(device)
        labels = labels.to(device)

        # Forward Propagation:
        # Generate 10 scores for each image.
        outputs = model(images)

        # argmax(dim=1):
        # Choose the digit with the highest score.
        predictions = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predictions == labels).sum().item()

        # Count the total number of test images.
        total += labels.size(0)

# Step 6: Calculate Accuracy
accuracy = 100 * correct / total

# Print important results
print("Test Images:", total)
print("Correct Predictions:", correct)
print(f"Test Accuracy: {accuracy:.2f}%")
