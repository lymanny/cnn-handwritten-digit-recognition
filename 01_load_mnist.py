
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Convert MNIST images into PyTorch tensors
transform = transforms.ToTensor()

# Load training dataset: 60,000 images
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Load testing dataset: 10,000 images
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Print dataset information
print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))

# Get the first image and its correct label
image, label = train_dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)

# Display the first 10 handwritten digit images
plt.figure(figsize=(12, 3))

for i in range(10):
    image, label = train_dataset[i]

    plt.subplot(1, 10, i + 1)
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title(str(label))
    plt.axis("off")

plt.tight_layout()
plt.show()
