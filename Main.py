# Import all the important libraries required for the moodel 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Load the dataset

data = pd.read_csv('data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']


# Display the first few rows of the dataset

print("Dataset Preview:")
print(data.head())


# Display dataset information

print("\nDataset Information:")
print(data.info)


# Pairplot to visualize relationships between variables

sns.pairplot(data, hue='auth')
plt.title('Pairplot of Features')
plt.show()

# Distribution of the target variable

plt.figure(figsize=(8, 6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'],palette=['blue', 'orange'])
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

# Logistic Regression Model

clf = LogisticRegression(solver='lbfgs', random_state=42)
clf.fit(X_train, y_train.values.ravel())

# Model Evaluation

y_pred = np.array(clf.predict(X_test))
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=["Pred.Negative", "Pred.Positive"],
                        index=['Act.Negative', "Act.Positive"])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = round((tn+tp)/(tn+fp+fn+tp), 4)
print(conf_mat)
print(f'\n Accuracy = {round(100*accuracy, 2)}%')


# Make predictions on new data

# Input The Details about the New Banknote in the form of 2D Array

new_banknote = np.array([[0,0,12.7957,-3.1353]])  # this set of values will determine the quality of the note

new_banknote = scalar.transform(new_banknote) 
prediction = clf.predict(new_banknote)[0]  
probability = clf.predict_proba(new_banknote)[0]  

# Print The Class which determines that the note is Fake(class = 1) or Real(class = 0)
print("\nPrediction:")
print(f'Class {prediction}')
print("\nProbability [0/1]:")
print(probability)

