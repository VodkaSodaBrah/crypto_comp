import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Debugging: Print XGBoost version and file path
print(f"XGBoost Version: {xgb.__version__}")
print(f"XGBoost File: {xgb.__file__}")

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBClassifier
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
)

# Train with early stopping
try:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=True,
    )
    print("Training successful!")
except TypeError as e:
    print(f"TypeError encountered: {e}")