
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Preprocess MNIST Images
transform = transforms.Compose([
    # Convert images into PyTorch tensors
    # Pixel values: 0–255 → 0–1
    transforms.ToTensor(),

    # Normalize pixel values using MNIST statistics
    # Mean = 0.1307, Standard Deviation = 0.3081
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 2: Load MNIST Datasets
# Training dataset: 60,000 handwritten digit images
train_dataset = datasets.MNIST(
    root="data",        # Folder to store the dataset
    train=True,         # Load training images
    download=True,      # Download if not available
    transform=transform # Apply preprocessing
)

# Testing dataset: 10,000 handwritten digit images
test_dataset = datasets.MNIST(
    root="data",
    train=False,        # Load testing images
    download=True,
    transform=transform
)

# Step 3: Create DataLoaders
# DataLoader divides images into smaller groups (batches)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,      # Process 32 images per batch
    shuffle=True        # Randomize training image order
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,      # Process 32 images per batch
    shuffle=False       # Keep testing image order
)

# Step 4: Get One Batch
# iter() creates an iterator over the DataLoader
# next() retrieves the first batch
images, labels = next(iter(train_loader))

# images: 32 normalized MNIST images
# labels: 32 correct digit labels (0–9)

# Print important information
print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))
print("Batch Shape:", images.shape)
print("Labels Shape:", labels.shape)

# Step 5: Visualize 8 Images
plt.figure(figsize=(10, 3))

for i in range(8):
    # Get one image from the batch
    image = images[i].squeeze()

    # Reverse normalization to display original pixel values
    # Normalized → original scale [0, 1]
    image = image * 0.3081 + 0.1307

    # Display the image and its correct label
    plt.subplot(1, 8, i + 1)
    plt.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    plt.title(str(labels[i].item()))
    plt.axis("off")

plt.tight_layout()
plt.show()
