# src/manage_model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
import os
import time

def manage_model_lifecycle():
    """
    Manages the lifecycle of the registered IrisLogisticRegressionModel
    by transitioning its stage and loading it.
    """
    # MLflow по умолчанию будет использовать локальную папку mlruns/ для трекинга.
    # Поэтому явная установка mlflow.set_tracking_uri() не требуется для локальной работы.

    client = MlflowClient()
    model_name = "IrisLogisticRegressionModel"

    print(f"\n--- Managing Model: {model_name} ---")

    # Получаем информацию о последней версии модели, которая зарегистрирована.
    # Ищем среди всех возможных стадий, чтобы найти последнюю зарегистрированную версию.
    try:
        # Получаем все версии для данной модели и берем последнюю зарегистрированную
        all_versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"])
        if not all_versions:
            print(f"No versions found for model '{model_name}'. Please ensure src/train_model.py was run to register the model.")
            return

        latest_version = all_versions[0].version
        print(f"Found latest registered version: {latest_version}")

    except Exception as e:
        print(f"An error occurred while fetching model versions: {e}")
        print("Please ensure you have run src/train_model.py at least once to register the model.")
        return

    # 1. Переводим модель в стадию "Staging"
    print(f"Transitioning model '{model_name}' (Version {latest_version}) to 'Staging'...")
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=True # Переместить все другие версии в Staging в Archive
    )
    print(f"Model '{model_name}' (Version {latest_version}) transitioned to 'Staging'.")
    time.sleep(1) # Дадим MLflow немного времени обновиться

    # Проверяем стадию
    model_version_details = client.get_model_version(model_name, latest_version)
    print(f"Current stage of Version {latest_version}: {model_version_details.current_stage}")

    # 2. Переводим модель в стадию "Production"
    print(f"Transitioning model '{model_name}' (Version {latest_version}) to 'Production'...")
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True # Переместить все другие версии в Production в Archive
    )
    print(f"Model '{model_name}' (Version {latest_version}) transitioned to 'Production'.")
    time.sleep(1)

    # Проверяем стадию
    model_version_details = client.get_model_version(model_name, latest_version)
    print(f"Current stage of Version {latest_version}: {model_version_details.current_stage}")

    # 3. Загружаем модель из "Production" стадии
    print(f"\n--- Loading Model from Production Stage ---")
    try:
        # URI для загрузки модели из Model Registry по имени и стадии
        model_uri = f"models:/{model_name}/Production"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from production: {model_uri}")

        # Проверяем, что модель работает
        import numpy as np
        # Пример данных для предсказания (первая строка из ирисов)
        # Эти значения - sepal_length, sepal_width, petal_length, petal_width
        sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = loaded_model.predict(sample_input)
        print(f"Prediction for sample input {sample_input[0]}: {prediction[0]}")

    except Exception as e:
        print(f"Error loading model from production: {e}")
        print("Ensure the model is correctly registered and transitioned to 'Production' stage.")

if __name__ == "__main__":
    manage_model_lifecycle()