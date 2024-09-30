# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Split the dataset into features (X) and target (y), and preprocess the data.
2. Split data into training and testing sets.
3. Train the Decision Tree Classifier using the training data.
4. Predict and evaluate the model on the test data, then visualize the decision tre
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: muralitharan k m
RegisterNumber:  212223040121
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
file_path = 'Employee.csv'
data = pd.read_csv(file_path)

# Preprocessing: Convert categorical variables to numerical
data = pd.get_dummies(data)

# Split features and target ('left' is the churn indicator)
X = data.drop('left', axis=1)
y = data['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()

```

## Output:
**![image](https://github.com/user-attachments/assets/69678c68-1fd3-4817-b000-c7f3e2c940c4)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
