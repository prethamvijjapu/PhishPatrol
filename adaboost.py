import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)
import joblib
import optuna

# Load and preprocess data
df = pd.read_csv('dataset2.csv').drop('index', axis=1)

# Convert target to binary (0/1)
df['Result'] = df['Result'].replace({-1: 0, 1: 1})

# Split features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1.0)

    # Create an AdaBoost model with the suggested hyperparameters
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Optuna minimizes the objective, so we return 1 - accuracy
    return 1 - accuracy

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Number of trials for optimization

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
best_model = AdaBoostClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    random_state=42
)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Save the best model
joblib.dump(best_model, 'optimized_adaboost_model.pkl')
print("Optimized AdaBoost model saved as 'optimized_adaboost_model.pkl'")

# Get feature importances
feature_importance = best_model.feature_importances_
feature_importance_dict = dict(zip(X.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 3 most important features:")
for feature, importance in sorted_features[:3]:
    print(f"{feature}: {importance:.4f}")
