# FER2013 Emotion Recognition 

**Made by Leith Engazzou, Hiba Chabbouh, Takoua Ayadi, Sahbi Ben Kalia and Yassine Dahmoul**

Overview
- Small project for facial emotion recognition using FER2013-style data.
- Includes preprocessing utilities, classical ML training, and DL checkpoint testing.

Repository structure
- `dataset/` : raw dataset folders (train/test) organized by emotion labels.
- `preprocessed_samples/` : preprocessed images (ignored by git).
- `saved_models/` : saved classical ML models (`.pkl`).
- `*.pth`, `*.pt` : deep-learning checkpoints present in project root.
- `streamlit_app.py` : Streamlit UI for testing saved `.pkl` ML models and `.pth` DL checkpoints.
- `ml_train.py` : helper for training/saving ML models (LogisticRegression, RF, SVM).
- `preprocessing.py` : preprocessing transforms and dataset helpers.
- Notebooks (`*.ipynb`) : experiments and training notebooks.
 - `resultat_preprocessing_clean.py` : a small clean runner that invokes `preprocessing.py` and `ml_train.py` for preprocessing + ML training.

Requirements
- Install dependencies from `requirements.txt` (recommended in a virtualenv).

Quick start
1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix / macOS
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit tester for quick inference (web UI):

```bash
streamlit run streamlit_app.py
```

4. Run the lightweight preprocessing + ML runner (uses defaults):

```bash
python resultat_preprocessing_clean.py
```

Notes on training
- The main ML training helpers are in `ml_train.py`. They expect either a `fer2013.csv` file (pre-extracted features) or image folders under `dataset/`.
- Example programmatic invocation:

```bash
python - <<PY
from ml_train import run_ml_models
import preprocessing
run_ml_models(preprocessing, csv_path='fer2013.csv', pipeline='light')
PY
```

- Many experiments are done in the included notebooks — open those for model-specific training flows.

Models and checkpoints
- Classical ML models are saved as `.pkl` files into `saved_models/` by `ml_train.py`.
- Deep-learning weights (`.pth` / `.pt`) are kept in the project root in current state; loading is handled by `streamlit_app.py` using a robust loader.

Dataset format
- `dataset/train/<emotion>/*.png` and `dataset/test/<emotion>/*.png` — emotion folder names should match the labels used in `preprocessing.py`.

Suggestions
- Keep large binary files (models, datasets, preprocessed samples) out of version control; use cloud storage or Git LFS if you need versioning.