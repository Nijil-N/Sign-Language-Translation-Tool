import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Path to dataset (change this to your local path)
data_dir = "asl_alphabet_train"

# Get list of class folders
classes = sorted(os.listdir(data_dir))
label_map = {i: cls for i, cls in enumerate(classes)}
inv_label_map = {cls: i for i, cls in label_map.items()}

# Load images and labels
images = []
labels = []

print("Loading images (this may take a minute)...")
for label in classes:
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path)[:1000]:  # Limit to 1000 images per class
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize to 64x64
        images.append(img)
        labels.append(inv_label_map[label])

images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# Save model
model.save('saved_model/asl_alphabet_model.h5')
print("Model saved to 'saved_model/asl_alphabet_model.h5'")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
