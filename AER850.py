import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib

data = pd.read_csv("C:\Python\Project 1 Data.csv")

#Data Visualization
data.head()
data.info()
sns.pairplot(data, hue='Step')

plt.show()

#Correlation Analysis
data.corr()
sns.heatmap(data.corr(),annot=True)

plt.show()

#Classification Model
X = data[['X', 'Y', 'Z']]
y = data['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classification = RandomForestClassifier()
classification.fit(X_train, y_train)
y_pred = classification.predict(X_test)
classicfication_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report)

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Model Evaluation
joblib.dump(classification, "C:\Python\Project1savedata.xlsx")
loaded_model = joblib.load("C:\Python\Project1savedata.xlsx")
new_coordinates = [[9.375, 3.0625, 1.51],[6.995, 5.125, 0.3875],[0, 3.0625, 1.93],[9.4, 3, 1.8],[9.4, 3, 1.3]]
predictions = loaded_model.predict(new_coordinates)
print("Predicted Maintenance Steps:", predictions)