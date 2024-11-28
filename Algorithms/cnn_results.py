import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

def load_data(data_directory, annotation_file, interested_classes):
    annotations = pd.read_csv(annotation_file)
    images = []
    labels = []
    label_map = {}
    selected_labels = [label for label in annotations['label'] if label in interested_classes]

    for index, row in annotations.iterrows():
        if row['label'] in interested_classes:
            img_path = os.path.join(data_directory, row['filename'])
            img = cv2.imread(img_path)  # Read image
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, (224, 224))  # Resize images to (224, 224)
                images.append(img)
                label = row['label']
                if label not in label_map:
                    label_map[label] = len(label_map)
                labels.append(label_map[label])

    return images, labels, label_map

# Specify the 10 classes of interest
interested_classes = [
    'SOUTHERN DOGFACE', 'ADONIS', 'BROWN SIPROETA', 'MONARCH',
    'GREEN CELLED CATTLEHEART', 'CAIRNS BIRDWING', 'EASTERN DAPPLE WHITE',
    'RED POSTMAN', 'MANGROVE SKIPPER', 'BLACK HAIRSTREAK'
]

# Load training dataset
X_train, y_train, label_map_train = load_data('Dataset/train', 'Dataset/Training_set.csv', interested_classes)

# Load testing dataset
X_test, y_test, label_map_test = load_data('Dataset/test', 'Dataset/Testing_set.csv', interested_classes)

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Load the saved CNN model
model = tf.keras.models.load_model('custom_cnn_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
y_pred_prob = model.predict(np.array(X_test))
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(np.array(y_test), axis=1)

# Filter the data to include only the instances of the interested classes
filtered_indices = [i for i, label in enumerate(y_true) if label in label_map_test.values()]
y_true_filtered = y_true[filtered_indices]
y_pred_filtered = y_pred[filtered_indices]

# Create class names for the filtered classes
class_names = [label for label, index in label_map_test.items() if index in label_map_test.values()]

# Print classification report
report = classification_report(y_true_filtered, y_pred_filtered, target_names=class_names)
print(report)

# Save the test results to a text file
with open('test_results.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\n")
    f.write(f"Test accuracy: {test_accuracy}\n")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(label_map_test.values()))
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Custom CNN')
plt.show()

# Plotting precision, recall, and F1-score for each class
precision = precision_score(y_true_filtered, y_pred_filtered, average=None)
recall = recall_score(y_true_filtered, y_pred_filtered, average=None)
f1 = f1_score(y_true_filtered, y_pred_filtered, average=None)

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
