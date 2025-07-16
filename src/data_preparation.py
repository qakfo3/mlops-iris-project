# src/data_preparation.py
import pandas as pd
from sklearn.datasets import load_iris
import os

def fetch_and_save_data(output_path="data/iris.csv"):
    """Fetches the Iris dataset and saves it as a CSV."""
    iris = load_iris(as_frame=True)
    df = iris.frame

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Iris dataset saved to {output_path}")

if __name__ == "__main__":
    fetch_and_save_data()