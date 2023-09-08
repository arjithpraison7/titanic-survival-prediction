# Titanic Survival Prediction (Machine Learning)

# Description

Predicting Titanic survival is a classic machine learning task that involves analyzing passenger data to predict whether a passenger survived or not.

## Libraries Used

- sci-kit learn
- pandas
- numpy

## Toolkit Used

- Jupyter Notebook

## Problem Statement

Given a dataset containing information about Titanic passengers, including attributes like age, gender, class, fare, and more, the task is to build a machine learning model that can predict whether a passenger survived the disaster or not.

The problem is a binary classification task, where the target variable is whether a passenger is labeled as "Survived" (1) or "Not Survived" (0).

## Data Collection

There is a lot of open-source data available on the Internet about the details of the Titanic survivors. We for this project choose the Kaggle data - [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)

### About the dataset

We have two main data sets `train.csv` and `test.csv`. Both of these data sets include a variety of features such as passenger information, ticket details, cabin information, etc. The `train.csv` includes the Survived (1 or 0) information about the passenger as well while the test.csv does not include that.

The following table shows the information that is available to us in these two data sets and their meaning:

 

| Variable | Definition | Key |
| --- | --- | --- |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex |  |
| Age | Age in years |  |
| sibsp | # of siblings / spouses aboard the Titanic |  |
| parch | # of parents / children aboard the Titanic |  |
| ticket | Ticket number |  |
| fare | Passenger fare |  |
| cabin | Cabin number |  |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

## Steps

1. Understand the Problem Statement
2. Gather the Dataset .
3. Preprocess the Data
4. Explore the Data
5. Choose a classifier
6. Training the classifier
7. Evaluation the model
8. Tuning Hyperparameters
9. Visualization

## Data Preprocessing

Preprocessing the data is a critical step in machine learning projects. It involves cleaning, transforming, and preparing the data to be suitable for training a model. The data we have in our hand at present seems to be entirely complete with a few missing values in the `age`, `cabin` columns.

Since "Age" is an important feature, it's essential to handle missing values carefully. One common approach is to impute missing ages with the median age of the dataset. The median is less sensitive to outliers compared to the mean. Here we replace the median value for the missing data using the following code:

```python
import pandas as pd

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Calculate median age from the training dataset
median_age = train_df['Age'].median()

# Impute missing ages in the train and test datasets
train_df['Age'].fillna(median_age, inplace=True)
test_df['Age'].fillna(median_age, inplace=True)
```

### Handling Categorical Variables

Encoding Sex and Embarked

The "Sex" and "Embarked" features are categorical variables. So we are dropping one of the binary columns to avoid redundancy using the following lines of code

```python
# One-hot encode "Sex" and "Embarked" features
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
```

### Feature Engineering

Creating FamilySize and IsAlone. A "FamilySize" feature by summing "SibSp" and "Parch" columns.

IsAlone feature is created whenever a passenger is traveling solo.

```python
# Create "FamilySize" and "IsAlone" features
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
train_df['IsAlone'] = (train_df['FamilySize'] == 0).astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['IsAlone'] = (test_df['FamilySize'] == 0).astype(int)
```

### Scaling/Normalization

Scaling helps features with different ranges contribute equally to the model's performance. We use standard scaling (z-score Normalization) for this approach.

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale numerical features
num_features = ['Fare', 'FamilySize']
train_df[num_features] = scaler.fit_transform(train_df[num_features])
test_df[num_features] = scaler.transform(test_df[num_features])
```

### Exploratory Data Analysis

We perform summary statistics, create visualizations, and perform correlation analysis using Python and popular data analysis libraries such as pandas, matplotlib, and seaborn:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Summary Statistics
summary_stats = train_df.describe()
print("Summary Statistics:\n", summary_stats)

# Exclude non-numeric columns from the DataFrame
numeric_columns = train_df.select_dtypes(include=['number']).columns
correlation_matrix = train_df[numeric_columns].corr()

# Visualization
# Count of survivors based on passenger class
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival Count by Passenger Class')
plt.show()

# Age distribution by survival status
plt.figure(figsize=(10, 6))
sns.histplot(x='Age', hue='Survived', data=train_df, kde=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

The output graphs and plots are shown in the images bellow:

![AgeDistribution.png](Titanic%20Survival%20Prediction%20(Machine%20Learning)%2047dd554ce73a4c6fac6bb26fa095cf37/AgeDistribution.png)

![SurvivalvsClass.png](Titanic%20Survival%20Prediction%20(Machine%20Learning)%2047dd554ce73a4c6fac6bb26fa095cf37/SurvivalvsClass.png)

![CorrelationMatrix.png](https://github.com/arjithpraison7/titanic-survival-prediction/blob/main/CorrelationMatrix.png)

## Choosing a classifier

Model selection is crucial in any machine learning project. In this project we go with the K-Nearest Neighbors (KNN) approach. It is simple and effective in many scenarios. The model is implemented as given bellow:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=5 (you can experiment with different values)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
```

```python
OUTPUT:
Accuracy: 0.553072625698324
Classification Report:
              precision    recall  f1-score   support

           0       0.59      0.78      0.67       105
           1       0.42      0.23      0.30        74

    accuracy                           0.55       179
   macro avg       0.51      0.51      0.49       179
weighted avg       0.52      0.55      0.52       179
```

Let's break down what each metric means:

1. **Accuracy:** This is the proportion of correctly classified instances out of the total instances. In your case, it's approximately 55.31%, which means that the model got about 55.31% of the predictions correct.
2. **Precision:** Precision is the proportion of true positive predictions (i.e., the number of correctly predicted "1"s) out of all instances predicted as positive. For class 0, it's about 59%, and for class 1, it's about 42%.
3. **Recall (Sensitivity or True Positive Rate):** Recall is the proportion of true positive predictions out of all actual positive instances. For class 0, it's about 78%, and for class 1, it's about 23%.
4. **F1-Score:** The F1-score is the harmonic mean of precision and recall. It provides a balanced measure between precision and recall. A high F1-score indicates a good balance between precision and recall.
5. **Support:** The number of occurrences of each class in the true dataset. For class 0, it's 105, and for class 1, it's 74.
6. **Macro Avg:** The average of precision, recall, and F1-score for both classes. It gives equal weight to both classes, regardless of class imbalance.
7. **Weighted Avg:** The weighted average of precision, recall, and F1-score, where the weights are the support values. This metric takes class imbalance into account.

## Hyperparameter Tuning

```python
# Define the parameter grid for k
param_grid = {'n_neighbors': range(1, 21)}

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameter
best_k = grid_search.best_params_['n_neighbors']

print(f"The best value of k is: {best_k}")

# Use this best value of k to train your final model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
```

In this code, we perform a grid search over the hyperparameter *k* using the range from 1 to 20. The best value of *k* is then used to train the final KNN model. Make sure you have your features and target variables correctly defined (`X_train` and `y_train`) before running this code.

# Conclusion

**Conclusion and Key Takeaways:**

1. **Data Preprocessing:**
    - Missing values in the 'Age' column were filled with the median age.
    - Categorical variables like 'Sex' and 'Embarked' were one-hot encoded.
2. **Feature Engineering:**
    - Created new features 'FamilySize' and 'IsAlone' based on the 'SibSp' and 'Parch' columns.
3. **Modeling:**
    - K-Nearest Neighbors (KNN) algorithm was used for classification.
    - The model was fine-tuned using GridSearchCV, resulting in an improved accuracy of 57.54%.
4. **Performance Evaluation:**
    - Before fine-tuning, the model had an accuracy of 55.31% with an F1-score of 52%.
    - After fine-tuning, the accuracy improved to 57.54%, although the model still struggles with classifying the 'Survived' class.
5. **Next Steps:**
    - Explore different models and algorithms, including ensemble methods and deep learning models.
    - Further feature engineering or selection may enhance model performance.
    - Evaluate the impact of additional external data or domain-specific knowledge.
6. **Considerations:**
    - Interpretability: KNN provides limited interpretability compared to some other models.
    - Class Imbalance: If the dataset is imbalanced, techniques like oversampling or using different evaluation metrics may be necessary.

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
Germany
