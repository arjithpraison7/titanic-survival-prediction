import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV


# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop non-numeric columns
train_df = train_df.select_dtypes(include=['number'])
test_df = test_df.select_dtypes(include=['number'])

# Fill missing values
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Assuming you have a target variable 'Survived'
y_train = train_df['Survived']
X_train = train_df.drop('Survived', axis=1)

X_test = test_df

# Calculate median age from the training dataset
median_age = X_train['Age'].median()

# Create "FamilySize" and "IsAlone" features
X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch']
X_train['IsAlone'] = (X_train['FamilySize'] == 0).astype(int)

X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch']
X_test['IsAlone'] = (X_test['FamilySize'] == 0).astype(int)

# Select only numeric columns
numeric_columns = X_train.select_dtypes(include=['number']).columns
X_train = X_train[numeric_columns]
X_test = X_test[numeric_columns]

# Initialize the scaler
scaler = StandardScaler()

# Scale numerical features
num_features = ['Fare', 'FamilySize']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Summary Statistics
summary_stats = X_train.describe()
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

y_pred = final_knn.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

final_predictions = final_knn.predict(X_test)

print("After finetuning :")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

