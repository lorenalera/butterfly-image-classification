import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

def load_data(data_directory, annotation_file, max_classes=10):
    annotations = pd.read_csv(annotation_file)
    images = []
    labels = []
    label_map = {}
    label_count = Counter()

    for index, row in annotations.iterrows():
        img_path = os.path.join(data_directory, row['filename'])
        img = cv2.imread(img_path)  # Read image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (224, 224))  # Resize images to (224, 224)
            label = row['label']
            if label not in label_map and len(label_map) < max_classes:
                label_map[label] = len(label_map)
            if label in label_map:
                images.append(img)
                labels.append(label_map[label])
                label_count[label_map[label]] += 1
                if len(label_count) >= max_classes and all(v >= 50 for v in label_count.values()):  # Ensure at least 50 images per class
                    break

    return images, labels, label_map

# Load training dataset with only the first 10 classes
X_train, y_train, label_map_train = load_data('Dataset/train', 'Dataset/Training_set.csv', max_classes=10)

# Load testing dataset with only the first 10 classes
X_test, y_test, label_map_test = load_data('Dataset/test', 'Dataset/Testing_set.csv', max_classes=10)

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Define a custom CNN model
def create_custom_cnn(input_shape, num_classes):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())  # Add batch normalization
    model.add(Activation('relu'))  # Activation function
    model.add(MaxPooling2D((2, 2)))  # Max pooling layer

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Fourth convolutional layer
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the output of the final pooling layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

    # Output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    return model

input_shape = (224, 224, 3)
num_classes_train = len(label_map_train)

model = create_custom_cnn(input_shape, num_classes_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(np.array(X_train), np.array(y_train), epochs=50, validation_split=0.2, batch_size=32)

# Save the model
model.save('custom_cnn_model.h5')

# Load the model
model = tf.keras.models.load_model('custom_cnn_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
y_pred_prob = model.predict(np.array(X_test))
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(np.array(y_test), axis=1)

# Print classification report
class_names = list(label_map_train.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save the test results to a text file
with open('test_results.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\n")
    f.write(f"Test accuracy: {test_accuracy}\n")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Custom CNN')
plt.show()

# Plotting precision, recall, and F1-score for each class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

plt.figure(figsize=(10, 6))
plt.plot(class_names, precision, marker='o', label='Precision')
plt.plot(class_names, recall, marker='s', label='Recall')
plt.plot(class_names, f1, marker='^', label='F1-Score')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
