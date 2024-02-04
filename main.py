import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import Bunch
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

# Function to visualize data
def visualize_data(dataset, class_column, dataset_name):
    sns.set(style="whitegrid")
    for column in dataset.columns[:-1]:  # Exclude the class column
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=class_column, y=column, data=dataset)
        plt.title(f"{column} distribution by {class_column} - {dataset_name}")
        plt.show()

# Function to apply K-nearest neighbors (KNN)
def apply_knn(dataset, class_column, dataset_name):
    X = dataset.drop(class_column, axis=1)
    y = dataset[class_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy for {dataset_name}: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to apply K-nearest neighbors (KNN) with 10-fold stratified cross-validation
def cross_validate_knn(dataset, class_column):
    X = dataset.drop(class_column, axis=1)
    y = dataset[class_column]

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Lists to store results
    all_true_labels = []
    all_predicted_probas = []
    accuracies = []
    classification_reports = []

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        knn.fit(X_train, y_train)

        # Predictions for precision-recall curve
        y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Probability of positive class
        all_true_labels.extend(y_test)
        all_predicted_probas.extend(y_pred_proba)

        # Predictions for accuracy and classification report
        y_pred = knn.predict(X_test)

        # Evaluate the model for accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Evaluate the model for classification report
        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        classification_reports.append(classification_report_dict)

        # Plot Precision-Recall curve for each fold
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'Fold {len(all_true_labels) // len(set(all_true_labels))}')

    # Plot Precision-Recall curve across all folds
    overall_precision, overall_recall, _ = precision_recall_curve(all_true_labels, all_predicted_probas)
    plt.plot(overall_recall, overall_precision, label='Overall', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve per Fold and Overall')
    plt.legend()

    # Add caption explaining the curves
    plt.figtext(0.5, 0.01, 'The solid lines represent Precision-Recall curves for each fold, while the dashed line represents the overall curve.', ha='center', va='center')

    plt.show()

    # Calculate performance metrics
    avg_precision = average_precision_score(all_true_labels, all_predicted_probas)
    avg_accuracy = np.mean(accuracies)  # Using numpy to calculate the mean
    avg_f1_score = f1_score(all_true_labels, [1 if p > 0.5 else 0 for p in all_predicted_probas])
    avg_precision_score = precision_score(all_true_labels, [1 if p > 0.5 else 0 for p in all_predicted_probas])
    avg_recall_score = recall_score(all_true_labels, [1 if p > 0.5 else 0 for p in all_predicted_probas])

    print("\nAverage Precision:", avg_precision)
    print("Average Accuracy:", avg_accuracy)
    print("Average F1-Score:", avg_f1_score)
    print("Average Precision Score:", avg_precision_score)
    print("Average Recall Score:", avg_recall_score)

    # Calculate average classification report
    unique_labels_list = unique_labels(all_true_labels)
    avg_classification_report = {label: [] for label in unique_labels_list}
    
    for class_report in classification_reports:
        for label in unique_labels_list:
            if label in class_report:
                avg_classification_report[label].append(class_report[label])

    for label in avg_classification_report:
        if len(avg_classification_report[label]) > 0:
            avg_classification_report[label] = np.mean(avg_classification_report[label])  # Using numpy to calculate the mean
        else:
            avg_classification_report[label] = 0  # Handle division by zero

    print("\nAverage Classification Report:\n", avg_classification_report)

# Ask for the file paths
file_path_A = input("Enter the path to A1_dataA.tsv: ").strip('"')
file_path_B = input("Enter the path to A1_dataB.tsv: ").strip('"')

# Load datasets
dataA = pd.read_csv(file_path_A, delimiter='\t')
dataB = pd.read_csv(file_path_B, delimiter='\t')

# Visualize datasets A and B
visualize_data(dataA, 'class', 'dataA')
visualize_data(dataB, 'class', 'dataB')

# Apply KNN on both datasets
print("\nKNN on Dataset A:")
apply_knn(dataA, 'class', 'dataA')

print("\nKNN on Dataset B:")
apply_knn(dataB, 'class', 'dataB')

# Perform 10-fold stratified cross-validation for both datasets
print("\nKNN 10-Fold Stratified Cross-Validation for Dataset A:")
cross_validate_knn(dataA, 'class')

print("\nKNN 10-Fold Stratified Cross-Validation for Dataset B:")
cross_validate_knn(dataB, 'class')