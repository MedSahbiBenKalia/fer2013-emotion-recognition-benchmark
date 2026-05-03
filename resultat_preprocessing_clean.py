"""
Clean runner that uses `preprocessing.py` and `ml_train.py`.
This file omits DL training and focuses on preprocessing + ML training.
"""
import os
from preprocessing import EMOTION_LABELS
import preprocessing as prep
import ml_train


def main(csv_path=None, data_root=None, pipeline="light"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = csv_path or os.path.join(script_dir, "fer2013.csv")
    data_root = data_root or os.path.join(script_dir, "dataset")

    results = ml_train.run_ml_models(prep, csv_path=csv_path, data_root=data_root, pipeline=pipeline)
    print("ML results:", results)


if __name__ == "__main__":
    main()
