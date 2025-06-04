import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

def add_sp_noise(img, salt_prob=0.01, pepper_prob=0.01):
    noisy_img = np.copy(img)
    total_pixels = img.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    salt_coords = [np.random.randint(0, i-1, num_salt) for i in img.shape]
    noisy_img[salt_coords[0], salt_coords[1], :] = 1 

    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in img.shape]
    noisy_img[pepper_coords[0], pepper_coords[1], :] = 0  

    return noisy_img

def adjust_brightness(img, factor=0.3):
    return np.clip(img * factor, 0, 1)

def apply_noises_and_light_conditions(img, salt_prob=0.05, pepper_prob=0.05, light_factor=0.7):
    img_with_noise = add_sp_noise(img, salt_prob, pepper_prob) 
    img_with_light_condition = adjust_brightness(img_with_noise, light_factor)  
    return img_with_light_condition

dataset_path = r"C:\Users\wsnlab\Desktop\Sahil\kagglehub\datasets\prasunroy\natural-images\versions\1\natural_images"
image_size = (64, 64)  

def load_images_from_directory(directory, image_size=(64, 64)):
    images, labels = [], []
    label_dict = {}  
    label_counter = 0

    for class_name in sorted(os.listdir(directory)):  
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            label_dict[label_counter] = class_name 
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img) / 255.0  
                images.append(img_array)
                labels.append(label_counter)  
            label_counter += 1  

    return np.array(images), np.array(labels), label_dict

all_images, all_labels, label_dict = load_images_from_directory(dataset_path)

x_train, x_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)
x_train_noisy_and_light = np.array([apply_noises_and_light_conditions(img) for img in x_train])
x_test_noisy_and_light = np.array([apply_noises_and_light_conditions(img) for img in x_test])

def build_autoencoder():
    input_img = layers.Input(shape=(64, 64, 3))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded_img = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded_img)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

autoencoder = build_autoencoder()
autoencoder.summary()
autoencoder.fit(x_train_noisy_and_light, x_train, epochs=30, batch_size=64, validation_data=(x_test_noisy_and_light, x_test))

cleaned_x_train = autoencoder.predict(x_train_noisy_and_light)
cleaned_x_test = autoencoder.predict(x_test_noisy_and_light)

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_dict), activation='softmax')  # Multi-class classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()
cnn_model.summary()

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

cnn_model.fit(datagen.flow(cleaned_x_train, y_train, batch_size=64), epochs=30, validation_data=(cleaned_x_test, y_test), callbacks=[early_stopping, reduce_lr])

test_loss, test_acc = cnn_model.evaluate(cleaned_x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

num_random_images = 5
random_indices = np.random.choice(len(x_test), num_random_images, replace=False)

for idx in random_indices:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(x_test[idx])
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(x_test_noisy_and_light[idx])
    plt.title("Noisy & Low Light Image")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cleaned_x_test[idx])
    plt.title("Denoised Image")
    plt.axis('off')

    pred = np.argmax(cnn_model.predict(np.expand_dims(cleaned_x_test[idx], axis=0)))  # Predict class
    label_name = label_dict.get(pred, "Unknown")

    plt.subplot(1, 4, 4)
    plt.imshow(cleaned_x_test[idx])
    plt.title(f"Predicted: {label_name}")
    plt.axis('off')

    plt.show()
