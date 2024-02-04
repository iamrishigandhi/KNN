import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import Bunch
from sklearn.utils.multiclass import unique_labels

# Function to apply K-nearest neighbors (KNN) with 10-fold stratified cross-validation
def cross_validate_knn(dataset, class_column):
    X = dataset.drop(class_column, axis=1)
    y = dataset[class_column]

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Lists to store results
    accuracies = []
    classification_reports = []

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        knn.fit(X_train, y_train)

        # Predictions
        y_pred = knn.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        classification_reports.append(classification_report_dict)

    # Print average accuracy and classification report over all folds
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("\nAverage Accuracy:", avg_accuracy)

    # Calculate average classification report
    unique_labels_list = unique_labels(y)
    avg_classification_report = {label: [] for label in unique_labels_list}
    
    for class_report in classification_reports:
        for label in unique_labels_list:
            if label in class_report:
                avg_classification_report[label].append(class_report[label])

    for label in avg_classification_report:
        if len(avg_classification_report[label]) > 0:
            avg_classification_report[label] = sum(avg_classification_report[label]) / len(avg_classification_report[label])
        else:
            avg_classification_report[label] = 0  # Handle division by zero

    print("\nAverage Classification Report:\n", avg_classification_report)

# Ask the user for the file paths
file_path_A = input("Enter the path to A1_dataA.tsv: ").strip('"')
file_path_B = input("Enter the path to A1_dataB.tsv: ").strip('"')

# Load datasets
dataA = pd.read_csv(file_path_A, delimiter='\t')
dataB = pd.read_csv(file_path_B, delimiter='\t')

# Perform 10-fold stratified cross-validation for both datasets
print("\nKNN 10-Fold Stratified Cross-Validation for Dataset A:")
cross_validate_knn(dataA, 'class')

print("\nKNN 10-Fold Stratified Cross-Validation for Dataset B:")
cross_validate_knn(dataB, 'class')