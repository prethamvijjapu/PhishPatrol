import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler
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

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    hidden_layer_sizes = tuple(trial.suggest_int(f'n_units_l{i}', 10, 100) for i in range(trial.suggest_int('n_layers', 1, 3)))
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1)
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    # Create an MLP model with the suggested hyperparameters
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_test_scaled)
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
best_model = MLPClassifier(
    hidden_layer_sizes=tuple(best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])),
    alpha=best_params['alpha'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=best_params['max_iter'],
    random_state=42
)
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate and display accuracy
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


# Save the best model and scaler
joblib.dump(best_model, 'optimized_mlp_model.pkl')
joblib.dump(scaler, 'mlp_scaler.pkl')
print("Optimized MLP model saved as 'optimized_mlp_model.pkl'")
print("Scaler saved as 'mlp_scaler.pkl'")
