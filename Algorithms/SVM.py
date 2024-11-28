import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for model saving

# Load the dataset from a CSV file
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

    return images, labels


# CSV file containing file paths and labels
train_csv_file = 'Dataset/Training_set.csv'
test_csv_file = 'Dataset/Testing_set.csv'
# Folder paths for the images
train_folder = 'Dataset/train'
test_folder = 'Dataset/test'

# Load images and labels from CSV files for training and test sets
train_images, train_labels = load_data(train_folder, train_csv_file)
test_images, test_labels = load_data(test_folder, test_csv_file)

# Preprocess images
def preprocess_images(images):
    # Resize images to a fixed size (e.g., 224x224)
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    # Flatten images
    flattened_images = [img.flatten() for img in resized_images]
    return np.array(flattened_images)

X_train = preprocess_images(train_images)
y_train = np.array(train_labels)
X_test = preprocess_images(test_images)
y_test = np.array(test_labels)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Evaluate SVM with linear kernel
svc = SVC(kernel='linear')
svc.fit(X_train_reduced, y_train)
y_pred = svc.predict(X_test_reduced)

print("Kernel: linear")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate accuracy, precision, recall, and F1-score for each class
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
accuracy = accuracy_score(y_test, y_pred)

# Extract scores for each class
class_names = np.unique(y_test)
precision_10 = precision[:10]
recall_10 = recall[:10]
f1_10 = f1[:10]

# Plot histograms for accuracy, precision, recall, and F1-score
plt.figure(figsize=(15, 10))

# Precision histogram
plt.subplot(2, 2, 1)
plt.bar(class_names[:10], precision_10)
plt.title('Precision for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('Precision')

# Recall histogram
plt.subplot(2, 2, 2)
plt.bar(class_names[:10], recall_10)
plt.title('Recall for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('Recall')

# F1-score histogram
plt.subplot(2, 2, 3)
plt.bar(class_names[:10], f1_10)
plt.title('F1-Score for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('F1-Score')

# Accuracy
plt.subplot(2, 2, 4)
plt.bar("Accuracy", accuracy)
plt.title('Accuracy')
plt.xlabel('Metric')
plt.ylabel('Score')

plt.tight_layout()
plt.show()
