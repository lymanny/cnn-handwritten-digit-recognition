
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# CNN Prediction (Inference):
# Uses a trained CNN to recognize handwritten digits.
# It does not update the model's weights.

# Step 1: Apply the same preprocessing as training
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert pixels to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 2: Load MNIST Test Images
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Step 3: Define the SAME CNN Architecture as training
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract 16 and 32 Feature Maps
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Activation and Pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convert features into 10 digit scores
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # Forward Propagation: Calculate digit scores
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


# Step 4: Load trained weights
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)

# eval(): Switch to prediction/evaluation mode
model.eval()

# Step 5: Predict 10 Test Images
plt.figure(figsize=(12, 4))

with torch.inference_mode():
    for i in range(10):
        # Get one image and its correct label
        image, label = test_dataset[i]

        # unsqueeze(0): Add batch dimension
        # [1, 28, 28] -> [1, 1, 28, 28]
        input_image = image.unsqueeze(0).to(device)

        # Run Forward Propagation
        output = model(input_image)

        # argmax: Select digit with highest score
        prediction = output.argmax(dim=1).item()

        # Reverse normalization for visualization
        display_image = image.squeeze() * 0.3081 + 0.1307

        # Display image with actual and predicted labels
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            display_image.numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(f"Actual: {label}\nPred: {prediction}")
        plt.axis("off")

        print(f"Image {i + 1}: Actual={label}, Predicted={prediction}")

plt.tight_layout()
plt.show()
