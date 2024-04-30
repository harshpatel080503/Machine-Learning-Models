import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
iris_df = pd.read_csv('E:/Data Science/Machine Learning Models/Machine Learning Models/Api/IRIS.csv')

# Assuming the last column is the target variable and all others are features
X = iris_df.iloc[:, :-1]  # All rows, all columns except the last one
y = iris_df.iloc[:, -1]   # All rows, only the last column

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model to disk
with open('E:/Data Science/Machine Learning Models/Machine Learning Models/Models/DecisionTree.pkl', 'wb') as f:
    pickle.dump(model, f)
