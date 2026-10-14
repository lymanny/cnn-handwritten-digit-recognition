
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


# ----------------------------------
# 1. CNN Model
# ----------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


# ----------------------------------
# 2. Load or Train CNN Model
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_mnist.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Train automatically when model file is missing
if not MODEL_PATH.exists():
    print("Trained model not found.")
    print("Training CNN automatically...")

    subprocess.run(
        [sys.executable, str(BASE_DIR / "05_train_cnn.py")],
        cwd=BASE_DIR,
        check=True
    )

# Create CNN and load trained weights
model = CNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("CNN model loaded successfully!")


# ----------------------------------
# 3. Drawing Canvas
# ----------------------------------
SIZE = 280
BRUSH_SIZE = 18

root = tk.Tk()
root.title("CNN Handwritten Digit Recognition")
root.resizable(False, False)

canvas = tk.Canvas(
    root,
    width=SIZE,
    height=SIZE,
    bg="black",
    cursor="cross",
    highlightthickness=0
)
canvas.pack(padx=20, pady=15)

# Store drawing for model prediction
drawing = Image.new(
    "L",
    (SIZE, SIZE),
    color=0
)

image_draw = ImageDraw.Draw(drawing)

last_point = None


# ----------------------------------
# 4. Draw Digits Using Mouse
# ----------------------------------
def draw_digit(event):
    global last_point

    x, y = event.x, event.y
    radius = BRUSH_SIZE // 2

    if last_point is not None:
        x0, y0 = last_point

        canvas.create_line(
            x0, y0, x, y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        image_draw.line(
            (x0, y0, x, y),
            fill=255,
            width=BRUSH_SIZE
        )

    canvas.create_oval(
        x - radius, y - radius,
        x + radius, y + radius,
        fill="white",
        outline="white"
    )

    image_draw.ellipse(
        (
            x - radius, y - radius,
            x + radius, y + radius
        ),
        fill=255
    )

    last_point = (x, y)


def stop_drawing(event):
    global last_point
    last_point = None


canvas.bind("<Button-1>", draw_digit)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_drawing)


# ----------------------------------
# 5. Preprocess Handwritten Digit
# ----------------------------------
def preprocess_digit():
    bbox = drawing.getbbox()

    if bbox is None:
        return None

    # Crop empty black background
    digit = drawing.crop(bbox)

    # Resize while preserving aspect ratio
    width, height = digit.size
    scale = 20 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Center digit in 28x28 image
    image = Image.new(
        "L",
        (28, 28),
        color=0
    )

    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    image.paste(digit, (left, top))

    # Convert to tensor and normalize like MNIST
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = (
        image_array - 0.1307
    ) / 0.3081

    image_tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0)

    return image_tensor.to(device)


# ----------------------------------
# 6. Predict Handwritten Digit
# ----------------------------------
def predict_digit():
    image_tensor = preprocess_digit()

    if image_tensor is None:
        result_label.config(
            text="Please draw a digit first!"
        )
        return

    with torch.inference_mode():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    digit = predicted.item()
    score = confidence.item() * 100

    result_label.config(
        text=f"Prediction: {digit}\nConfidence: {score:.2f}%"
    )


# ----------------------------------
# 7. Clear Drawing
# ----------------------------------
def clear_canvas():
    global drawing, image_draw, last_point

    canvas.delete("all")

    drawing = Image.new(
        "L",
        (SIZE, SIZE),
        color=0
    )

    image_draw = ImageDraw.Draw(drawing)
    last_point = None

    result_label.config(
        text="Draw a digit (0–9)"
    )


# ----------------------------------
# 8. Application Interface
# ----------------------------------
result_label = tk.Label(
    root,
    text="Draw a digit (0–9)",
    font=("Arial", 16)
)
result_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 12),
    width=12,
    command=predict_digit
)
predict_button.pack(
    side=tk.LEFT,
    padx=5
)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 12),
    width=12,
    command=clear_canvas
)
clear_button.pack(
    side=tk.LEFT,
    padx=5
)

root.mainloop()
