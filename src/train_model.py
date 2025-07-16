# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
import os

def train_model(data_path="data/iris.csv"):
    """
    Loads data, trains a Logistic Regression model, and logs results with MLflow.
    """
    # Удалена строка mlflow.set_tracking_uri() - MLflow будет использовать mlruns/ по умолчанию

    with mlflow.start_run():
        # 1. Загрузка данных
        try:
            df = pd.read_csv(data_path)
            print(f"Data loaded from {data_path}")
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Please ensure src/data_preparation.py was run.")
            return

        X = df.drop(columns=['target'])
        y = df['target']

        # 2. Разделение данных на обучающую и тестовую выборки
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # 3. Определение параметров модели
        solver = 'liblinear'
        max_iter = 100

        # Логирование параметров с MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)

        # 4. Обучение модели
        model = LogisticRegression(solver=solver, max_iter=max_iter)
        model.fit(X_train, y_train)
        print("Model trained.")

        # 5. Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        # Логирование метрик с MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")

        # 6. Сохранение (логирование) модели с MLflow
        mlflow.sklearn.log_model(model, "iris_model", registered_model_name="IrisLogisticRegressionModel")
        print("Model logged to MLflow.")

if __name__ == "__main__":
    train_model()