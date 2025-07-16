# student_dropout_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate dataset (normally this would be loaded from CSV or database)
np.random.seed(42)
num_samples = 500
data = {
    'login_frequency': np.random.poisson(lam=3, size=num_samples),
    'avg_session_duration': np.random.normal(loc=30, scale=10, size=num_samples),
    'assignments_submitted': np.random.randint(0, 10, num_samples),
    'forum_posts': np.random.randint(0, 15, num_samples),
    'attendance_rate': np.random.uniform(0.4, 1.0, size=num_samples),
    'dropout': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Features and target
X = df.drop('dropout', axis=1)
y = df['dropout']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Dropout Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
