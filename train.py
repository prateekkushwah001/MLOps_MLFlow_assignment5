import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

# Set the experiment name in MLflow
mlflow.set_experiment("Titanic Model Comparison Experiment")

# Load the Titanic dataset
data = sns.load_dataset('titanic')

# Feature engineering: Convert categorical variables to numeric and drop NaN
data = data[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

X = data[['pclass', 'sex', 'age', 'fare']]  # Features
y = data['survived']  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model and log metrics only
with mlflow.start_run(run_name="Linear_Regression"):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    predictions_lr = model_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, predictions_lr)
    r2_lr = r2_score(y_test, predictions_lr)

    # Log parameters and metrics (no model file)
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("mse", mse_lr)
    mlflow.log_metric("r2", r2_lr)
    print(f"Linear Regression - MSE: {mse_lr}, R-squared: {r2_lr}")

# Train Random Forest model and log metrics only
with mlflow.start_run(run_name="Random_Forest"):
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, predictions_rf)
    r2_rf = r2_score(y_test, predictions_rf)

    # Log parameters and metrics (no model file)
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse_rf)
    mlflow.log_metric("r2", r2_rf)
    print(f"Random Forest - MSE: {mse_rf}, R-squared: {r2_rf}")

