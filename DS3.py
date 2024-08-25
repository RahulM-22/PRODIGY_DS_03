import pandas as pd

# Load the dataset (replace with your path)
data_path = 'D:/MP 1 FINAL/DS 03/bank+marketing/bank-additional/bank-additional/bank-additional-full.csv'  # Replace with the actual path to the dataset
df = pd.read_csv(data_path, delimiter=';')

# Display the first few rows
print(df.head())
# Convert categorical variables into numerical values using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features and target variable
X = df_encoded.drop('y_yes', axis=1)  # 'y_yes' indicates whether a customer subscribed
y = df_encoded['y_yes']

# Display the preprocessed data
print(X.head())
from sklearn.model_selection import train_test_split

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.show()
