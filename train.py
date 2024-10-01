import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split dataset
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an input example (1 sample)
input_example = X_test.iloc[0].to_dict()

# Define and train Linear Regression model
def train_linear_regression():
    with mlflow.start_run(run_name="Linear_Regression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log parameters, metrics, and model with input example
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(lr, "model", input_example=input_example)

        print(f"Linear Regression MSE: {mse}")

# Define and train Random Forest model
def train_random_forest():
    with mlflow.start_run(run_name="Random_Forest"):
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log parameters, metrics, and model with input example
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(rf, "model", input_example=input_example)

        print(f"Random Forest MSE: {mse}")

if __name__ == "__main__":
    mlflow.set_experiment("MLflow_Experiment_Tracking")
    train_linear_regression()
    train_random_forest()

