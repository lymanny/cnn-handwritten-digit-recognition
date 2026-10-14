
import tkinter as tk

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

# Interactive CNN Prediction:
# Draw a digit using the mouse, preprocess it,
# and predict its class using a trained MNIST CNN.

# Step 1: Define the SAME CNN architecture as Lesson 5.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract image features.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: Remove negative activation values.
        self.relu = nn.ReLU()

        # Pooling: Reduce feature map dimensions.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten and output 10 digit scores.
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # Forward Propagation:
    # Convert an image into 10 digit scores.
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


# Step 2: Load the trained model from Lesson 5.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

# The file must exist in the project folder.
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)

# Evaluation mode: Do not train or update weights.
model.eval()


# Step 3: Create a drawing window.
SIZE = 280

root = tk.Tk()
root.title("CNN Handwritten Digit Recognition")
root.resizable(False, False)

# Canvas: Black background, white handwriting.
canvas = tk.Canvas(
    root,
    width=SIZE,
    height=SIZE,
    bg="black",
    cursor="cross"
)
canvas.pack(padx=20, pady=15)

# Pillow image:
# Store the same drawing in memory for prediction.
drawing = Image.new("L", (SIZE, SIZE), color=0)
painter = ImageDraw.Draw(drawing)

# Brush size for drawing digits.
BRUSH_SIZE = 18


# Step 4: Draw with the mouse.
def draw(event):
    x, y = event.x, event.y
    radius = BRUSH_SIZE // 2

    # Display a white circle on the canvas.
    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    # Save the same circle to the Pillow image.
    painter.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ),
        fill=255
    )


# Draw while holding the left mouse button.
canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-1>", draw)


# Step 5: Preprocess the handwritten digit.
def preprocess_image():
    # Find pixels belonging to the digit.
    box = drawing.getbbox()

    if box is None:
        return None

    # Crop empty black space around the digit.
    digit = drawing.crop(box)

    # Resize the digit while preserving its shape.
    width, height = digit.size
    scale = 20 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Center the digit on a 28x28 black image.
    image = Image.new("L", (28, 28), color=0)

    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    image.paste(digit, (left, top))

    # Convert pixel values from 0-255 to 0-1.
    array = np.array(image, dtype=np.float32) / 255.0

    # Apply the SAME normalization as MNIST training.
    array = (array - 0.1307) / 0.3081

    # Tensor shape: [Batch=1, Channel=1, 28, 28].
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


# Step 6: Predict the digit.
def predict():
    image_tensor = preprocess_image()

    if image_tensor is None:
        result_label.config(text="Please draw a digit first.")
        return

    # Inference: Calculate outputs without gradients.
    with torch.inference_mode():
        outputs = model(image_tensor)

        # Softmax: Convert raw scores into probabilities.
        probabilities = torch.softmax(outputs, dim=1)

        # Choose the digit with the highest probability.
        prediction = probabilities.argmax(dim=1).item()

        # Show the model's probability for its chosen class.
        confidence = probabilities[0, prediction].item() * 100

    result_label.config(
        text=f"Prediction: {prediction}\nConfidence: {confidence:.1f}%"
    )

    print(f"Predicted Digit: {prediction}")
    print(f"Model Confidence: {confidence:.1f}%")


# Step 7: Clear the drawing.
def clear():
    canvas.delete("all")

    # Reset Pillow drawing to black.
    painter.rectangle(
        (0, 0, SIZE, SIZE),
        fill=0
    )

    result_label.config(text="Draw a digit (0-9)")


# Step 8: Add buttons and prediction result.
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict,
    width=12
)
predict_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear,
    width=12
)
clear_button.pack(side=tk.LEFT, padx=5)

result_label = tk.Label(
    root,
    text="Draw a digit (0-9)",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=15)

# Start the drawing application.
root.mainloop()
