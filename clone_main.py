import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Load the dataset

data = pd.read_csv('data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']

# Display the first few rows of the dataset

print("Dataset Preview:")
print(data.head())

# Display dataset information

print("\nDataset Information:")
print(data.info())

# Pairplot to visualize relationships between variables

sns.pairplot(data, hue='auth')
plt.title('Pairplot of Features')
plt.show()

# Distribution of the target variable

plt.figure(figsize=(8, 6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'], palette=['blue', 'orange'])
target_count = data.auth.value_counts()

plt.annotate(text=str(target_count[0]), xy=(-0.04, 10 + target_count[0]), xytext=(-0.04, 10 + target_count[0]), size=14)
plt.annotate(text=str(target_count[1]), xy=(0.96, 10 + target_count[1]), xytext=(0.96, 10 + target_count[1]), size=14)
plt.ylim(0, 900)
plt.show()

# Balance the dataset

nb_to_delete = target_count[0] - target_count[1]
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
data = data[nb_to_delete:]
print("Balanced Target Counts:")
print(data['auth'].value_counts())

# Splitting the dataset into features (X) and target variable (y)

X = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']

# Splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling

scalar = StandardScaler()

scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Hyperparameter tuning using GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())

print("Best Parameters:", grid_search.best_params_)

# Cross-validation for model evaluation

cv_accuracy = cross_val_score(grid_search.best_estimator_, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')
cv_precision = cross_val_score(grid_search.best_estimator_, X_train, y_train.values.ravel(), cv=5, scoring='precision')
cv_recall = cross_val_score(grid_search.best_estimator_, X_train, y_train.values.ravel(), cv=5, scoring='recall')
cv_f1 = cross_val_score(grid_search.best_estimator_, X_train, y_train.values.ravel(), cv=5, scoring='f1')

print("Cross-Validation Results:")
print("CV Accuracy:", np.mean(cv_accuracy))
print("CV Precision:", np.mean(cv_precision))
print("CV Recall:", np.mean(cv_recall))
print("CV F1 Score:", np.mean(cv_f1))

# Model Evaluation on Test Set

y_pred = grid_search.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest Set Results:")
print("Confusion Matrix:")
print(pd.DataFrame(conf_mat, columns=["Pred.Negative", "Pred.Positive"], index=['Act.Negative', "Act.Positive"]))
print("\nClassification Report:")
print(classification_rep)
print("\nAccuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Make predictions on new data

# Input The Details about the New Banknote in the form of 2D Array

new_banknote = np.array([[10.0, -50.0, 10.0, 50]])  # this set of values will determine the quality of the note 
new_banknote = scalar.transform(new_banknote)
prediction = grid_search.predict(new_banknote)[0]
probability = grid_search.predict_proba(new_banknote)[0]

# Print The Class which determines that the note is Fake(class = 1) or Real(class = 1) 
print("\nPrediction:")
print(f'Class {prediction}')
print("\nProbability [0/1]:")
print(probability)


