from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("num_model.h5")

# Load and preprocess the image
img = Image.open("./handwrittennumbers/1b.png").convert("L")  # Convert to grayscale
img = img.resize((28, 28))  # Resize to MNIST format
img = np.array(img)
# img = 255 - img  # Invert (MNIST: white digit on black bg)
img = img / 255.0  # Normalize
img = img.reshape(1, 784)  # Flatten to (1, 784)

# Optional: Display the image
# import matplotlib.pyplot as plt
# plt.imshow(img.reshape(28, 28), cmap="gray")
# plt.title("Prepared image")
# plt.axis("off")
# plt.show()

# Predict
prediction = model.predict(img)
# predicted_class = np.argmax(prediction)
# print("Predicted digit:", predicted_class)

for i in range(10):
    print(f"Probability of {i}: {prediction[0][i]:.4f}")

# show image
import matplotlib.pyplot as plt
plt.imshow(img.reshape(28, 28), cmap="gray")
plt.title("Prepared image")
plt.axis("off")
plt.show()