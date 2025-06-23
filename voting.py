# Import all required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('dataset2.csv')

# -------------------
# Data Preprocessing
# -------------------
# Drop ID column
data = data.drop('index', axis=1)

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode target variable
le = LabelEncoder()
X = data.drop('Result', axis=1)
y = le.fit_transform(data['Result'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Model Definitions
# -------------------
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Voting': VotingClassifier(estimators=[
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ], voting='soft')
}

# -------------------
# Model Training & Evaluation
# -------------------
for name, model in models.items():
    print(f"\n{'='*40}\n{name} Classifier\n{'='*40}")

    if name == 'MLP':
        train_X, test_X = X_train_scaled, X_test_scaled
    else:
        train_X, test_X = X_train, X_test

    model.fit(train_X, y_train)

    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Features:")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        print(importances.sort_values(ascending=False).head(5))

    print(f"\nSample Prediction Probabilities:")
    print(pd.DataFrame(y_proba[:5], columns=le.classes_, index=[f"Sample {i+1}" for i in range(5)]))

    print(f"\n{'='*40}\n")

# -------------------
# Additional Information
# -------------------
print("\nCLASS LABEL MAPPING:")
for i, class_name in enumerate(le.classes_):
    print(f"{i} -> {class_name}")
