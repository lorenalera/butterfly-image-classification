# import os
# from collections import Counter
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# import joblib
# import pandas as pd
#
# def load_images_from_csv(csv_file, folder):
#     df = pd.read_csv(csv_file)
#     images = []
#     labels = []
#     for index, row in df.iterrows():
#         img_path = os.path.join(folder, row['filename'])
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
#         if img is not None:
#             images.append(img)
#             labels.append(row['label'])
#     return images, labels
#
# def preprocess_images(images):
#     # Resize images to a fixed size (e.g., 224x224)
#     resized_images = [cv2.resize(img, (224, 224)) for img in images]
#     # Flatten images
#     flattened_images = [img.flatten() for img in resized_images]
#     return np.array(flattened_images)
#
# def knn_classify(X_train, y_train, X_test, k=3):
#     predictions = []
#     for test_instance in X_test:
#         distances = []
#         for train_instance, train_label in zip(X_train, y_train):
#             distance = euclidean_distance(test_instance, train_instance)
#             distances.append((distance, train_label))
#         distances.sort(key=lambda x: x[0])
#         nearest_neighbors = distances[:k]
#         nearest_labels = [label for _, label in nearest_neighbors]
#         most_common_label = Counter(nearest_labels).most_common(1)[0][0]
#         predictions.append(most_common_label)
#     return predictions
#
# def euclidean_distance(point1, point2):
#     return np.sqrt(np.sum((point1 - point2) ** 2))
#
# def save_knn_model(model, file_path):
#     joblib.dump(model, file_path)
#     print(f'Model saved to {file_path}')
#
# def load_knn_model(file_path):
#     return joblib.load(file_path)
#
# def plot_confusion_matrix(cm, class_labels):
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title('Confusion Matrix')
#     plt.show()
#
# def plot_classification_report(cr, class_labels):
#     cr_df = pd.DataFrame(cr).iloc[:-1, :10].T  # Take the first 10 classes
#     cr_df.plot(kind='bar', figsize=(14, 7))
#     plt.title('Classification Report Metrics')
#     plt.ylabel('Scores')
#     plt.xlabel('Classes')
#     plt.legend(loc='upper right')
#     plt.xticks(rotation=45)
#     plt.show()
#
# # Load and preprocess data
# train_images, train_labels = load_images_from_csv('Dataset/Training_set.csv', 'Dataset/train')
# test_images, test_labels = load_images_from_csv('Dataset/Testing_set.csv', 'Dataset/test')
#
# X_train = preprocess_images(train_images)
# y_train = np.array(train_labels)
# X_test = preprocess_images(test_images)
# y_test = np.array(test_labels)
#
# # Train and save KNN model
# k = 5
# y_pred = knn_classify(X_train, y_train, X_test, k)
# save_knn_model((X_train, y_train, k), 'knn_model.joblib')
#
# # Evaluate model
# print(f"KNN with k={k}")
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# cr = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)], output_dict=True)
# print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)]))
#
# # Plot confusion matrix
# plot_confusion_matrix(cm, [f'Class {i}' for i in range(10)])
#
# # Plot classification report metrics
# plot_classification_report(cr, [f'Class {i}' for i in range(10)])
import os
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd

def load_images_from_csv(csv_file, folder):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        if img is not None:
            images.append(img)
            labels.append(row['label'])
    return images, labels

def preprocess_images(images):
    # Resize images to a fixed size (e.g., 224x224)
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    # Flatten images
    flattened_images = [img.flatten() for img in resized_images]
    return np.array(flattened_images)

def knn_classify(X_train, y_train, X_test, k=3):
    predictions = []
    for test_instance in X_test:
        distances = []
        for train_instance, train_label in zip(X_train, y_train):
            distance = euclidean_distance(test_instance, train_instance)
            distances.append((distance, train_label))
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k]
        nearest_labels = [label for _, label in nearest_neighbors]
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    return predictions

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def save_knn_model(X_train, y_train, k, file_path):
    joblib.dump((X_train, y_train, k), file_path)
    print(f'Model saved to {file_path}')

def load_knn_model(file_path):
    return joblib.load(file_path)

def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(cr, class_labels):
    cr_df = pd.DataFrame(cr).iloc[:-1, :].T  # Take the first 10 classes
    cr_df.plot(kind='bar', figsize=(14, 7))
    plt.title('Classification Report Metrics')
    plt.ylabel('Scores')
    plt.xlabel('Classes')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

# Load and preprocess data
train_images, train_labels = load_images_from_csv('Dataset/Training_set.csv', 'Dataset/train')
test_images, test_labels = load_images_from_csv('Dataset/Testing_set.csv', 'Dataset/test')

# Convert to numpy arrays
X_train = preprocess_images(train_images)
X_test = preprocess_images(test_images)

# Filter training data to include only the first 10 classes
unique_classes = np.unique(train_labels)[:10]
filtered_indices_train = [i for i, label in enumerate(train_labels) if label in unique_classes]
X_train_filtered = X_train[filtered_indices_train]
y_train_filtered = np.array(train_labels)[filtered_indices_train]

# Filter test data to include only the first 10 classes
filtered_indices_test = [i for i, label in enumerate(test_labels) if label in unique_classes]
X_test_filtered = X_test[filtered_indices_test]
y_test_filtered = np.array(test_labels)[filtered_indices_test]

# Train KNN model
k = 3
save_knn_model(X_train_filtered, y_train_filtered, k, 'knn_model.joblib')

# Load KNN model
X_train_loaded, y_train_loaded, k_loaded = load_knn_model('knn_model.joblib')

# Perform predictions on the filtered test data
y_pred_filtered = knn_classify(X_train_loaded, y_train_loaded, X_test_filtered, k_loaded)

# Evaluate model on the first 10 classes
print(f"KNN with k={k_loaded} on first 10 classes")
cm_filtered = confusion_matrix(y_test_filtered, y_pred_filtered)
print(cm_filtered)
cr_filtered = classification_report(y_test_filtered, y_pred_filtered, target_names=[f'Class {i}' for i in unique_classes], output_dict=True)
print(classification_report(y_test_filtered, y_pred_filtered, target_names=[f'Class {i}' for i in unique_classes]))

# Plot confusion matrix
plot_confusion_matrix(cm_filtered, [f'Class {i}' for i in unique_classes])

# Plot classification report metrics
plot_classification_report(cr_filtered, [f'Class {i}' for i in unique_classes])
