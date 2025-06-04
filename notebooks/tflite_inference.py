import os
import time
import logging
import numpy as np
import psutil
import tensorflow as tf
from PIL import Image
from memory_profiler import memory_usage

# Suppress TensorFlow and XNNPACK logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define Paths
AUTOENCODER_TFLITE = "autoencoder.tflite"
CNN_TFLITE = "cnn_model.tflite"
INPUT_IMAGE_PATH = "test2.jpg"

# Helper function to get storage size
def get_storage_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

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
    num_cores = psutil.cpu_count()
    normalized_cpu = ((end_cpu + start_cpu) / 2) / num_cores  # Normalized to 0-100% of total capacity
    return mem_usage, normalized_cpu, exec_time, result

# Function: Preprocess Image
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# TFLite inference for autoencoder
def autoencoder_inference(interpreter, input_image):
    interpreter.set_tensor(autoencoder_input_details[0]['index'], input_image)
    interpreter.invoke()
    return interpreter.get_tensor(autoencoder_output_details[0]['index'])

# TFLite inference for CNN
def cnn_inference(interpreter, input_image):
    interpreter.set_tensor(cnn_input_details[0]['index'], input_image)
    interpreter.invoke()
    return interpreter.get_tensor(cnn_output_details[0]['index'])

# Load the TFLite Models
autoencoder_interpreter = tf.lite.Interpreter(model_path=AUTOENCODER_TFLITE)
autoencoder_interpreter.allocate_tensors()
cnn_interpreter = tf.lite.Interpreter(model_path=CNN_TFLITE)
cnn_interpreter.allocate_tensors()

# Get input/output details
autoencoder_input_details = autoencoder_interpreter.get_input_details()
autoencoder_output_details = autoencoder_interpreter.get_output_details()
cnn_input_details = cnn_interpreter.get_input_details()
cnn_output_details = cnn_interpreter.get_output_details()

# Print storage
print("TFLite Storage (MB):")
print(f"AUTOENCODER: {get_storage_size(AUTOENCODER_TFLITE):.2f}")
print(f"CNN: {get_storage_size(CNN_TFLITE):.2f}")

# Load Image
input_image = preprocess_image(INPUT_IMAGE_PATH)

# Warm-up runs
for _ in range(3):
    autoencoder_inference(autoencoder_interpreter, input_image)
    denoised_image = autoencoder_inference(autoencoder_interpreter, input_image)
    denoised_image = denoised_image.astype(np.float32) / 255.0
    cnn_inference(cnn_interpreter, denoised_image)

# Measure inference
# Step 1: Autoencoder
auto_mem, auto_cpu, auto_time, denoised_image = measure_inference(
    autoencoder_inference, autoencoder_interpreter, input_image
)
print(f"\nAutoencoder Inference Metrics (RAM in MB, CPU %, Execution Time in ms):")
print(f"RAM={auto_mem:.2f}, CPU={auto_cpu:.2f}%, Time={auto_time:.2f}ms")

# Convert output back to image
denoised_image = (denoised_image[0] * 255).astype(np.uint8)
denoised_pil = Image.fromarray(denoised_image)
denoised_pil.save("denoised_output_tflite.jpg")
print("? Denoised image saved: denoised_output_tflite.jpg")

# Step 2: CNN
if len(denoised_image.shape) > 4:
    denoised_image = np.squeeze(denoised_image, axis=0)  # Remove extra batch dimension if needed
denoised_image = denoised_image.astype(np.float32) / 255.0
denoised_image = np.expand_dims(denoised_image, axis=0)  # Add single batch dimension
cnn_mem, cnn_cpu, cnn_time, predictions = measure_inference(
    cnn_inference, cnn_interpreter, denoised_image
)
print(f"\nCNN Inference Metrics (RAM in MB, CPU %, Execution Time in ms):")
print(f"RAM={cnn_mem:.2f}, CPU={cnn_cpu:.2f}%, Time={cnn_time:.2f}ms")

# Get predicted class
predicted_class = np.argmax(predictions)
label_dict = {0: "arm", 1: "elbow", 2: "face", 3: "foot", 4: "hand", 5: "leg", 6: "random"}
predicted_label = label_dict.get(predicted_class, "Unknown")
print(f"? Predicted Class: {predicted_label} (Class {predicted_class})")