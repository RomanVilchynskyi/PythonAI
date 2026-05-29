# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # === Step 1: Load model ===
# model = load_model("num_cnn_model.h5")

# # === Step 2: Load custom image ===
# img_dir = "./handwrittennumbers"

# # Get all image file paths
# image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# for image_file in sorted(image_files):  # Sorted for consistent order

#     img_path = os.path.join(img_dir, image_file)
#     # Load, convert to grayscale, resize to 28x28
#     img = Image.open(img_path).convert("L").resize((28, 28))
#     img_array = np.array(img, dtype="float32")

#     # Invert if filename starts with 'w'
#     if image_file.lower().startswith("w"):
#         img_array = 255.0 - img_array

#     # Normalize and reshape to match input shape (1, 28, 28, 1)
#     img_array = (img_array / 255.0).reshape(1, 28, 28, 1)

#     # === Step 3: Predict ===
#     prediction = model.predict(img_array, verbose=0)
#     predicted_label = np.argmax(prediction)

#     # === Step 4: Show result ===
#     plt.imshow(img_array.squeeze(), cmap="gray")
#     plt.title(f"Predicted Digit: {predicted_label}")
#     plt.axis("off")
#     plt.show()



from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

model = load_model("num_cnn_model.h5", compile=False)

img_dir = "./handwrittennumbers"

image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print("Знайдено файлів:", len(image_files))
print(image_files)

for image_file in sorted(image_files):
    img_path = os.path.join(img_dir, image_file)

    img = Image.open(img_path).convert("L").resize((28, 28))
    img_array = np.array(img, dtype="float32")

    if image_file.lower().startswith("w"):
        img_array = 255.0 - img_array

    img_array = (img_array / 255.0).reshape(1, 28, 28, 1)

    print("Обробляю:", image_file)

    prediction = model.predict(img_array, verbose=0)
    predicted_label = np.argmax(prediction)

    print(f"{image_file} -> {predicted_label}")

    plt.figure()
    plt.imshow(img_array.squeeze(), cmap="gray")
    plt.title(f"{image_file} -> {predicted_label}")
    plt.axis("off")

# показати все в кінці
plt.show()