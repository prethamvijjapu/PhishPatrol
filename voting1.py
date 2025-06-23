import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import optuna
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load and preprocess data
df = pd.read_csv('dataset.csv').drop('index', axis=1)
df['Result'] = df['Result'].replace({-1: 0, 1: 1})

# Split features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna optimization
def objective(trial):
    # Random Forest parameters
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('rf_max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
    }

    # XGBoost parameters
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-3, 0.5, log=True)
    }

    # MLP parameters with pipeline
    mlp_params = {
        'mlp__hidden_layer_sizes': tuple(
            trial.suggest_int(f'mlp_units_layer{i}', 50, 150) 
            for i in range(trial.suggest_int('mlp_n_layers', 1, 3))
        ),
        'mlp__alpha': trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
        'mlp__learning_rate_init': trial.suggest_float('mlp_learning_rate', 1e-4, 0.1, log=True)
    }

    # Create models
    rf = RandomForestClassifier(**rf_params, random_state=42)
    xgb = XGBClassifier(**xgb_params, random_state=42)
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=42))
    ]).set_params(**mlp_params)

    # Voting classifier
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('mlp', mlp)],
        voting='soft'
    )

    # Train and evaluate
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params

# Build final model with best parameters
best_rf = RandomForestClassifier(
    n_estimators=best_params['rf_n_estimators'],
    max_depth=best_params['rf_max_depth'],
    min_samples_split=best_params['rf_min_samples_split'],
    random_state=42
)

best_xgb = XGBClassifier(
    n_estimators=best_params['xgb_n_estimators'],
    max_depth=best_params['xgb_max_depth'],
    learning_rate=best_params['xgb_learning_rate'],
    random_state=42
)

best_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=tuple(
            best_params[f'mlp_units_layer{i}'] 
            for i in range(best_params['mlp_n_layers'])
        ),
        alpha=best_params['mlp_alpha'],
        learning_rate_init=best_params['mlp_learning_rate'],
        max_iter=1000,
        random_state=42
    ))
])

# Final voting classifier
final_voting = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb), ('mlp', best_mlp)],
    voting='soft'
)

# Train final model
final_voting.fit(X_train, y_train)

# Make predictions
y_pred = final_voting.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)  # Calculate ROC-AUC score

# Print evaluation metrics
print("\nFinal Model Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")  # Print ROC-AUC score

# Plot ROC curve for the VotingClassifier
fpr, tpr, thresholds = roc_curve(y_test, final_voting.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'VotingClassifier ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# ROC curves for individual models
for name, model in final_voting.estimators_:
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)  # For models like SVM
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} ROC curve (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Individual Models')
plt.legend(loc="lower right")
plt.show()

# Feature importance from trained Random Forest
trained_rf = final_voting.estimators_[0]
feature_importance = trained_rf.feature_importances_
fi_dict = dict(zip(X.columns, feature_importance))
sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:3]

print("\nTop 3 Important Features:")
for feat, imp in sorted_fi:
    print(f"{feat}: {imp:.4f}")

# Save model
joblib.dump(final_voting, 'voting_classifier.pkl')