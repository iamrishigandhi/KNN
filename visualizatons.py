import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Ask the user for the file path
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