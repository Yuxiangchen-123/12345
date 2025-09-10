# TCRT Model Training & Testing

This project implements the **Temporal Causal Residual TCN (TCRT)** framework with:
- **DI-TCN**: Dynamic Inception Temporal Convolutional Network
- **GRCT Head**: Split Fusion Convolution, ConvGRU, and Multi-Head Attention

## Features
- Loads behavioral dataset (Excel format, e.g., `behavioral_dataset_92k.xlsx`).
- Converts daily activity logs into sequences.
- Trains the TCRT model with PyTorch.
- Evaluates with Accuracy, Precision, Recall, F1, AUC.
- Saves the best model checkpoint.

## Requirements
```bash
pip install torch torchvision torchaudio scikit-learn pandas openpyxl joblib
```

## Usage
1. Place your dataset Excel file (e.g., `behavioral_dataset_92k.xlsx`) in the project folder.
2. Run training and testing:

```bash
python tcrt_train_test.py --data behavioral_dataset_92k.xlsx --seq_len 30 --epochs 30 --batch_size 256 --lr 3e-4
```

## Outputs
- `tcrt_best.pt` – Best model checkpoint
- `label_encoder.json` – Label encoder classes
- `minmax_scaler.joblib` – Feature scaler for deployment

## Notes
- If your Excel file does not have `Employment_Status`, synthetic labels are generated for demonstration.
- To add real employment labels, include a column `Employment_Status` with values {0,1,2} or {Unemployed,Employed,Promoted}.
- You can merge external socioeconomic indicators (GDP, unemployment rate, etc.) into the dataset for richer features.

