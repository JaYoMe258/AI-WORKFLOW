# hospital_readmission_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate hospital data
np.random.seed(42)
data = {
    'age': np.random.randint(30, 90, 300),
    'days_in_hospital': np.random.randint(1, 20, 300),
    'num_procedures': np.random.randint(0, 5, 300),
    'prior_readmissions': np.random.randint(0, 3, 300),
    'has_chronic_conditions': np.random.choice([0, 1], 300),
    'readmitted': np.random.choice([0, 1], 300, p=[0.75, 0.25])
}
df = pd.DataFrame(data)

# Features and target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('Hospital Readmission – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
#
# Feature importance
feature_importances = clf.feature_importances_ 
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance for Hospital Readmission')
plt.tight_layout()
plt.show()