"""Test an MNIST-trained model on your own image of a digit.

Usage:
    python test_mnist.py my_digit.png
    python test_mnist.py my_digit.png my_model.h5
"""
import sys

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

image_path = sys.argv[1] if len(sys.argv) > 1 else "digit.png"
model_path = sys.argv[2] if len(sys.argv) > 2 else "mnist_model.h5"  # or "mnist_model.keras"

# 1. Load the trained model
model = load_model(model_path)

# 2. Load and preprocess the image so it matches the MNIST format:
#    28x28, grayscale, white digit on black background, values in [0, 1]
img = Image.open(image_path).convert("L")      # grayscale
img = img.resize((28, 28))

arr = np.array(img)

# MNIST digits are WHITE on BLACK. A typical photo/drawing is black on white,
# so invert it. If your image is already white-on-black, set INVERT = False.
INVERT = True
if INVERT:
    arr = 255 - arr

arr = arr.astype("float32") / 255.0            # normalize to [0, 1]

# 3. Reshape to what the model expects.
#    CNN:   (1, 28, 28, 1)
#    Dense: (1, 784)   <- use this line instead if your model starts with Flatten-less Dense layers
x = arr.reshape(1, 28, 28, 1)
# x = arr.reshape(1, 784)

# 4. Predict
probs = model.predict(x)[0]
pred = int(np.argmax(probs))

print(f"Predicted digit: {pred}  (confidence: {probs[pred]:.2%})")
print("All probabilities:")
for digit, p in enumerate(probs):
    print(f"  {digit}: {p:.4f}")

# Optional: view what the model actually saw
import matplotlib.pyplot as plt
plt.imshow(arr, cmap="gray")
plt.title(f"Prediction: {pred}")
plt.axis("off")
plt.show()
