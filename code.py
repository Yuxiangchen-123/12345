import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)             
        y = y.squeeze(-1)                 
        y = self.conv(y.unsqueeze(1))   
        y = self.sigmoid(y).squeeze(1)    
        y = y.unsqueeze(-1)               
        return x * y.expand_as(x)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding != 0 else out

class TCNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.ecalayer = ECALayer(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.resample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.ecalayer(out)
        if self.resample:
            res = self.resample(res)
        return self.relu(out + res)

class DI_TCN(nn.Module):
    def __init__(self, in_ch, base_ch=128, num_classes=4, depth=6):
        super(DI_TCN, self).__init__()
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_c = in_ch if i == 0 else base_ch
            layers.append(TCNResidualBlock(in_c, base_ch, dilation=dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(base_ch, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = torch.mean(out, dim=2)
        return self.fc(out)


class ActivityDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        self.label_encoder = LabelEncoder()
        df["Activity_Category"] = self.label_encoder.fit_transform(df["Activity_Category"])
        features = df[["Student_ID", "Day", "Month", "Year"]].values
        labels = df["Activity_Category"].values
        self.scaler = MinMaxScaler()
        features = self.scaler.fit_transform(features)
        self.sequences, self.labels = [], []
        for i in range(len(features) - seq_len):
            self.sequences.append(features[i:i + seq_len])
            self.labels.append(labels[i + seq_len])
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_and_evaluate(data_path, output_model_path, epochs=30, batch_size=64, test_size=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(data_path)
    dataset = ActivityDataset(df, seq_len=30)

    X, y = dataset.sequences, dataset.labels
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size, stratify=y, random_state=42)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

  
    class_sample_count = np.bincount(y[train_idx])
    weights = 1. / class_sample_count
    sample_weights = [weights[t] for t in y[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_ch = X.shape[2]
    num_classes = len(np.unique(y))
    model = DI_TCN(in_ch=in_ch, base_ch=128, num_classes=num_classes, depth=6).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(epochs):
        
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

       
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
        except:
            auc = 0.0

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
              f"F1: {f1:.4f} | AUC: {auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_model_path)
            print(f" Best model saved (Epoch {epoch+1}) with Acc {acc:.4f}")

    print(f"\n Final Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    data_path = r"path_of_the_dataset"
    output_model_path = r"path_of_the_model"
    train_and_evaluate(data_path, output_model_path, epochs=30, batch_size=64, test_size=0.2)
