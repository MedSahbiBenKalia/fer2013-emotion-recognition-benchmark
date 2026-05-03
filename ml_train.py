import os
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from preprocessing import EMOTION_LABELS


def _get_models_dir() -> str:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    models_dir = os.path.join(script_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_ml_model(model, model_name: str, pipeline: str):
    models_dir = _get_models_dir()
    safe_name = model_name.lower().replace(" ", "_")
    fname = f"{safe_name}_{pipeline}.pkl"
    fpath = os.path.join(models_dir, fname)
    joblib.dump(model, fpath)
    print(f"Saved ML model to: {fpath}")


def compute_metrics_and_print(y_true, y_pred, model_name: str):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"[{model_name}] Accuracy : {acc*100:.2f}%")
    print(f"[{model_name}] Precision: {prec*100:.2f}%")
    print(f"[{model_name}] Recall   : {rec*100:.2f}%")
    return {"accuracy": acc, "precision": prec, "recall": rec}


def plot_confusion_matrix_from_preds(y_true, y_pred, labels, model_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels,
           ylabel="Vrai label", xlabel="Label prédit", title=f"Matrice de confusion - {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    fig.tight_layout()
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model_name.replace(' ', '_')}_confusion_{ts}.png"
    fpath = os.path.join(results_dir, fname)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    print(f"Saved confusion matrix to: {fpath}")
    plt.close(fig)


def run_ml_models(preprocessing_module, csv_path: str = "fer2013.csv", data_root: str = "dataset", pipeline: str = "light"):
    # preprocessing_module is the imported preprocessing module
    ml_results = {}
    X_train_flat = X_test_flat = None
    y_train_labels = y_test_labels = None

    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        X, y = preprocessing_module.preprocess_fer2013_dataframe(df, pipeline=pipeline)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["emotion"])
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train.argmax(axis=1))
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_train_labels = y_train.argmax(axis=1)
        y_test_labels = y_test.argmax(axis=1)
    else:
        X_train_flat, X_test_flat, y_train_labels, y_test_labels = preprocessing_module.load_ml_data_from_image_folders(data_root, pipeline=pipeline)

    # Logistic Regression
    print("\n[ML] Logistic Regression")
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=800, tol=3e-3, solver="saga", random_state=42, n_jobs=-1)
    lr_model.fit(X_train_flat, y_train_labels)
    save_ml_model(lr_model, "logistic_regression", pipeline)
    lr_pred = lr_model.predict(X_test_flat)
    ml_results["Logistic Regression"] = {"accuracy": accuracy_score(y_test_labels, lr_pred), "time": time.time() - t0}
    ml_results["Logistic Regression"].update(compute_metrics_and_print(y_test_labels, lr_pred, "Logistic Regression"))

    # Random Forest
    print("\n[ML] Random Forest")
    t0 = time.time()
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train_labels)
    save_ml_model(rf_model, "random_forest", pipeline)
    rf_pred = rf_model.predict(X_test_flat)
    ml_results["Random Forest"] = {"accuracy": accuracy_score(y_test_labels, rf_pred), "time": time.time() - t0}
    ml_results["Random Forest"].update(compute_metrics_and_print(y_test_labels, rf_pred, "Random Forest"))

    # SVM (subset)
    print("\n[ML] SVM (subset)")
    t0 = time.time()
    subset_size = min(8000, len(X_train_flat))
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    svm_model = SVC(kernel="rbf", C=10, gamma='scale', random_state=42, verbose=False)
    svm_model.fit(X_train_flat[indices], y_train_labels[indices])
    save_ml_model(svm_model, "svm", pipeline)
    svm_pred = svm_model.predict(X_test_flat)
    ml_results["SVM"] = {"accuracy": accuracy_score(y_test_labels, svm_pred), "time": time.time() - t0}
    ml_results["SVM"].update(compute_metrics_and_print(y_test_labels, svm_pred, "SVM"))

    return ml_results
