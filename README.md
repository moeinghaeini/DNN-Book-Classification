## Book Cover Classifier (EfficientNetB0 + Hyperparameter Tuning)

A Colab-first project for training an image classifier that predicts the category of a book from its cover. It downloads data from Kaggle, fetches cover images from URLs, trains a transfer-learning model with KerasTuner, evaluates on a held-out test set, generates rich visualizations, and exports deployable artifacts.

### Key features
- **Automated data acquisition**: pulls the Kaggle dataset `mohamedbakhet/amazon-books-reviews` and downloads images from the provided URLs
- **Transfer learning**: EfficientNetB0 backbone with optional fine‑tuning
- **Hyperparameter search**: KerasTuner RandomSearch for head size, dropout, learning rate, and fine‑tuning depth
- **Robust training**: class weighting, data augmentation, early stopping, learning‑rate scheduling, and checkpointing
- **Evaluation & insights**: accuracy, macro‑F1, classification report, confusion matrix, learning curves
- **Interpretability & exploration**: misclassification gallery, t‑SNE/UMAP embeddings, Grad‑CAM overlays
- **Exportable model**: SavedModel and `.keras` formats plus label mapping JSON

## Dataset
- **Source**: [Amazon Books Reviews (Kaggle)](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)
- **Columns used**:
  - `image`: URL to the book cover
  - `category`: target label
- **Filtering**: rows with missing URL/label are dropped; classes can be filtered by minimum frequency; optionally limit to top‑K classes

## Project structure
- `Book_Cover_Classifier_Colab.ipynb`: end‑to‑end notebook (data → training → evaluation → export)

## Requirements
- Recommended: Google Colab with GPU runtime enabled
- Python 3.9+ if running locally
- Core packages (installed automatically in the notebook):
  - `tensorflow`, `keras-tuner`, `scikit-learn`, `pandas`, `matplotlib`, `pillow`, `tqdm`, `opendatasets`, `umap-learn`, `kaggle`

### Local setup (optional)
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U tensorflow keras-tuner scikit-learn pandas matplotlib pillow tqdm opendatasets umap-learn kaggle

# (Optional) Launch Jupyter to run the notebook locally
pip install -U jupyterlab
jupyter lab
```

### Kaggle credentials
If `opendatasets` cannot download automatically, the notebook falls back to Kaggle CLI.
- Create a token at `https://www.kaggle.com/` → Account → Create New Token
- In Colab, the notebook will prompt for `username` and `key`
- Locally, place `kaggle.json` at `~/.kaggle/kaggle.json` with permissions `600`

## Quickstart (Colab)
1. Open `Book_Cover_Classifier_Colab.ipynb` in Google Colab
2. Runtime → Change runtime type → set Hardware accelerator to GPU
3. Run cells from top to bottom. When prompted, provide Kaggle credentials (if needed)
4. The notebook will download data, train the model, evaluate, visualize, and export artifacts

## Configuration
Adjust these in the config cell of the notebook:
- **Data**: `CSV_PATH` (auto‑set after download), `URL_COL` (default `image`), `LABEL_COL` (default `category`), `DATA_DIR`
- **Sampling**: `MIN_SAMPLES_PER_CLASS`, `TOP_K_CLASSES`
- **Splits**: `VAL_FRAC`, `TEST_FRAC`, `SEED`
- **Images**: `IMG_SIZE`, `BATCH_SIZE`
- **Downloads**: `MAX_DOWNLOAD_WORKERS`, `RETRY_ATTEMPTS`, `TIMEOUT_SEC`, `USER_AGENT`

## Training pipeline
1. Load and clean the CSV; visualize label distribution
2. Download and cache images in parallel; drop failed downloads
3. Encode labels and create stratified train/val/test splits
4. Build `tf.data` pipelines with EfficientNet preprocessing and light augmentations
5. Compute class weights for imbalance
6. Hyperparameter search (RandomSearch) over head size, dropout, learning rate, and fine‑tuning position
7. Train final model with best hyperparameters and callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
8. Evaluate on the test set; compute metrics and plots
9. Visualize embeddings (t‑SNE/UMAP), misclassifications, and Grad‑CAM
10. Export model and label mappings

## Outputs
Artifacts are written under `DATA_DIR` (default `/content/book_covers/` in Colab):
- `export/model_savedmodel/`: TensorFlow SavedModel (serving‑ready)
- `export/model.keras`: Keras format
- `export/label_mapping.json`: `class_to_idx` and `idx_to_class` dictionaries

## Inference example (local)
```python
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Paths produced by the notebook (adjust if you changed DATA_DIR)
export_dir = "book_covers/export"
model = tf.keras.models.load_model(f"{export_dir}/model.keras")
with open(f"{export_dir}/label_mapping.json") as f:
    label_maps = json.load(f)
idx_to_class = {int(k): v for k, v in label_maps["idx_to_class"].items()}

IMG_SIZE = (224, 224)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)[None, ...]
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

x = preprocess("/path/to/cover.jpg")
probs = model.predict(x)
pred_idx = int(np.argmax(probs, axis=1)[0])
print(idx_to_class[pred_idx])
```

## Troubleshooting
- **Kaggle download fails**: ensure `kaggle.json` is present and has permissions `600`, or use the interactive prompt in Colab
- **Too many failed image downloads**: reduce `MAX_DOWNLOAD_WORKERS`, increase `TIMEOUT_SEC`, or rerun to leverage caching
- **CUDA/GPU issues locally**: prefer Colab GPU; otherwise install TensorFlow versions compatible with your CUDA/cuDNN stack
- **OOM during training**: lower `BATCH_SIZE`, reduce `IMG_SIZE`, or disable fine‑tuning

## Acknowledgements
- Dataset by Mohamed Bakhet on Kaggle
- EfficientNet (Tan & Le) via `tf.keras.applications`
- KerasTuner for hyperparameter search

### Results
- **Test accuracy**: 0.3972 (6 classes, 720 images)
- **Macro metrics**: precision 0.40, recall 0.40, F1 0.39
- **Best LR**: 0.001 (val_acc ≈ 0.414 during LR search)

```text
Classification report:
precision    recall  f1-score   support

['Biography & Autobiography']       0.42      0.36      0.39       120
     ['Business & Economics']       0.43      0.56      0.49       120
                  ['Fiction']       0.32      0.42      0.36       120
                  ['History']       0.36      0.24      0.29       120
         ['Juvenile Fiction']       0.63      0.57      0.60       120
                 ['Religion']       0.24      0.23      0.23       120

                     accuracy                           0.40       720
                    macro avg       0.40      0.40      0.39       720
                 weighted avg       0.40      0.40      0.39       720
```

- **Confusion matrix**:

![Confusion matrix](23_confusion_matrix_00.png)

- **Training/visualization figure**:

![Training curves / visualization](25_figure_00.png)

- **Sample prediction** (top-1 probabilities plotted for an example):

![Sample prediction](output.png)

## License
This project currently has no explicit license. Consider adding one (e.g., MIT, Apache‑2.0) to clarify usage and distribution.
