import os
import time
import numpy as np
import psutil
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# Helper function to get storage size
def get_storage_size(file_path):
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

# Preprocess image
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Helper function to measure inference
def measure_inference(func, *args):
    process = psutil.Process()
    start_time = time.time()
    start_cpu = process.cpu_percent(interval=None)
    mem_usage = memory_usage((func, args), interval=0.1, max_usage=True)
    result = func(*args)
    end_time = time.time()
    end_cpu = process.cpu_percent(interval=None)
    exec_time = (end_time - start_time) * 1000  # ms
    normalized_cpu = ((end_cpu + start_cpu) / 2) / psutil.cpu_count()  # Normalized to 0-100% of total capacity
    return mem_usage, normalized_cpu, exec_time, result

# Model and image paths
autoencoder_path = "autoencoder_model.onnx"
cnn_path = "cnn_model.onnx"
input_image_path = "test2.jpg"

# Label dictionary
label_dict = {0: "arm", 1: "elbow", 2: "face", 3: "foot", 4: "hand", 5: "leg", 6: "random"}

# Check storage
print("ONNX Storage (MB):")
print(f"AUTOENCODER: {get_storage_size(autoencoder_path):.2f}")
print(f"CNN: {get_storage_size(cnn_path):.2f}")
print()

# Load models
try:
    autoencoder_session = ort.InferenceSession(autoencoder_path, providers=['CPUExecutionProvider'])
    cnn_session = ort.InferenceSession(cnn_path, providers=['CPUExecutionProvider'])
    autoencoder_input_name = autoencoder_session.get_inputs()[0].name
    cnn_input_name = cnn_session.get_inputs()[0].name
except Exception as e:
    print(f"Failed to load models: {str(e)}")
    exit(1)

# Load image
try:
    input_image = preprocess_image(input_image_path)
except Exception as e:
    print(f"Failed to load image: {str(e)}")
    exit(1)

# Step 1: Denoising with Autoencoder
def autoencoder_inference():
    return autoencoder_session.run(None, {autoencoder_input_name: input_image})[0]

print("Autoencoder Inference Metrics (RAM in MB, CPU %, Execution Time in ms):")
try:
    auto_mem, auto_cpu, auto_time, denoised_image = measure_inference(autoencoder_inference)
    print(f"  RAM={auto_mem:.2f}, CPU={auto_cpu:.2f}%, Time={auto_time:.2f}ms")
except Exception as e:
    print(f"  Failed: {str(e)}")
    exit(1)
print()

# Convert output to image format
denoised_image = (denoised_image[0] * 255).astype(np.uint8)
denoised_pil = Image.fromarray(denoised_image)
denoised_pil.save("denoised_output_onnx.jpg")
print("? Denoised image saved: denoised_output_onnx.jpg")

# Step 2: Classification using CNN
def cnn_inference():
    denoised_input = denoised_image.astype(np.float32) / 255.0
    denoised_input = np.expand_dims(denoised_input, axis=0)
    return cnn_session.run(None, {cnn_input_name: denoised_input})[0]

print("CNN Inference Metrics (RAM in MB, CPU %, Execution Time in ms):")
try:
    cnn_mem, cnn_cpu, cnn_time, predictions = measure_inference(cnn_inference)
    print(f"  RAM={cnn_mem:.2f}, CPU={cnn_cpu:.2f}%, Time={cnn_time:.2f}ms")
except Exception as e:
    print(f"  Failed: {str(e)}")
    exit(1)
print()

# Get predicted class
predicted_class = np.argmax(predictions)
predicted_label = label_dict.get(predicted_class, "Unknown")
print(f"? Predicted Class: {predicted_label} (Class {predicted_class})")

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(input_image))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_pil)
plt.title("Denoised Image")
plt.axis('off')

display_text = "No person found" if predicted_label.lower() == "random" else f"Person - {predicted_label}"

plt.subplot(1, 3, 3)
plt.imshow(denoised_pil)
plt.title(display_text)
plt.axis('off')

plt.show()