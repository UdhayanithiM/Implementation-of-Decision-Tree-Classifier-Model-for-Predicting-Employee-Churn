# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Read the CSV file
data = pd.read_csv("/content/Employee_EX6.csv")

# Display the first few rows of the data
print(data.head())

# Get information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Check the distribution of the target variable
print(data["left"].value_counts())

# Use LabelEncoder to encode the 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features (X) and target variable (y)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(criterion="entropy")

# Train the classifier
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new data
prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print(prediction)

```

## Output:
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127933352/58d50643-3f9b-44b8-aac8-59d43ac02e81)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127933352/15273bc4-d129-4748-b7d5-4d30f890d1d2)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127933352/7b580597-a295-4cb2-9aef-f74e6b787e51)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127933352/84d6f667-d74d-48ad-affe-d580a4f5c586)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
