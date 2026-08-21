"""
main.py
-------
Runs the full pipeline end-to-end:
1. Preprocess raw data
2. Train the model
3. Evaluate it and save metrics/plot

Usage:
    python main.py
"""

from src.preprocessing import run_preprocessing
from src.train import train_model
from src.evaluate import evaluate_model


def run_pipeline():
    print("=" * 50)
    print("STEP 1: Preprocessing")
    print("=" * 50)
    run_preprocessing()

    print("\n" + "=" * 50)
    print("STEP 2: Training")
    print("=" * 50)
    pipeline, X_test, y_test = train_model()

    print("\n" + "=" * 50)
    print("STEP 3: Evaluation")
    print("=" * 50)
    evaluate_model(pipeline, X_test, y_test)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
