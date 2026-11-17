"""Base MLP baseline.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Base MLP baseline (dosage genotype only)")
    p.add_argument("--phe", type=str, default="HKW", help="Dataset name (e.g., HKW)")
    p.add_argument("--data_dir", type=str, default="", help="Data directory")
    p.add_argument("--result_dir", type=str, default="", help="Result directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dims", type=str, default="1024,512,256,128")
    return p.parse_args()


def _resolve_data_dir(args) -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    phe = args.phe.rstrip("/\\")
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(project_root, "data", "data_zeamap", phe)
    if not data_dir.endswith(os.sep):
        data_dir += os.sep
    args.data_dir = data_dir
    return data_dir


def _load_y(data_dir: str, phe: str) -> np.ndarray:
    y_path_root = os.path.join(data_dir, "yData.csv")
    y_path_legacy = os.path.join(data_dir, "data_temporary", "yData.csv")
    y_path = y_path_root if os.path.exists(y_path_root) else y_path_legacy
    y_df = pd.read_csv(y_path, sep=",", header=None)
    if y_df.shape[1] >= 2 and phe == "KW":
        return np.asarray(y_df.iloc[:, 1].values, dtype=np.float64).reshape(-1)
    return np.asarray(y_df.values, dtype=np.float64).reshape(-1)


def load_data(args) -> tuple[np.ndarray, np.ndarray]:
    """Load dosage genotype X (N, P) and phenotype y (N,)."""
    data_dir = _resolve_data_dir(args)
    phe = args.phe.rstrip("/\\")
    y = _load_y(data_dir, phe)

    dosage_path = os.path.join(data_dir, "genotype_dosage.npy")
    if not os.path.isfile(dosage_path):
        raise FileNotFoundError(
            f"Missing {dosage_path}. Expected (N, P) 0/1/2 dosage for Base_MLP."
        )
    X = np.asarray(np.load(dosage_path), dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"genotype_dosage.npy must be 2D (N, P), got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != y rows {y.shape[0]}")
    print(f"[{phe}] Base_MLP dosage X={X.shape} y={y.shape}")
    return X, y


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout_rate: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EarlyStopping:
    def __init__(self, patience: int = 20, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> None:
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pcc = 0.0
    else:
        pcc, _ = pearsonr(y_true, y_pred)
        if not np.isfinite(pcc):
            pcc = 0.0
    return mae, mse, r2, float(pcc)


def train_and_evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    X, y = load_data(args)
    phe = args.phe.rstrip("/\\")
    if not args.result_dir:
        args.result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "result", "Base_MLP", phe
        )
    os.makedirs(args.result_dir, exist_ok=True)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_pccs, fold_maes = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold} ===")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_va = scaler.transform(X[val_idx]).astype(np.float32)

        model = MLP(X.shape[1], hidden_dims=hidden_dims, dropout_rate=args.dropout).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        early = EarlyStopping(patience=args.early_stop_patience)

        X_train = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
        X_val = torch.tensor(X_va, dtype=torch.float32, device=device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

        best_pcc, best_sd = -float("inf"), None
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(X_train.size(0), device=device)
            for i in range(0, X_train.size(0), args.batch_size):
                bi = perm[i : i + args.batch_size]
                opt.zero_grad()
                pred = model(X_train[bi])
                loss = criterion(pred, y_train[bi].view(-1, 1))
                loss.backward()
                opt.step()

            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds.append(model(xb).detach().cpu())
                    labels.append(yb.detach().cpu())
            val_pred = torch.cat(preds).numpy()
            val_true = torch.cat(labels).numpy()
            mae, mse, r2, pcc = _regression_metrics(val_true, val_pred)
            print(
                f"Epoch {epoch}, Val PCC: {pcc:.4f}, MAE: {mae:.4f}, "
                f"MSE: {mse:.4f}, R2: {r2:.4f}"
            )
            early(pcc)
            if pcc > best_pcc:
                best_pcc = pcc
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if early.early_stop:
                print("Early stopping triggered.")
                break

        if best_sd is not None:
            model.load_state_dict(best_sd)
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).detach().cpu())
                labels.append(yb.detach().cpu())
        val_pred = torch.cat(preds).numpy().reshape(-1)
        val_true = torch.cat(labels).numpy().reshape(-1)
        mae_f, mse_f, r2_f, pcc_f = _regression_metrics(val_true, val_pred)
        print(f"Fold {fold} best: PCC={pcc_f:.4f}, MAE={mae_f:.4f}")
        fold_pccs.append(pcc_f)
        fold_maes.append(mae_f)

        torch.save(
            {
                "model_state_dict": best_sd,
                "input": "dosage",
                "input_dim": X.shape[1],
                "hidden_dims": hidden_dims,
                "dropout": args.dropout,
                "fold": fold,
                "pcc": float(pcc_f),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            os.path.join(args.result_dir, f"all_best_model_fold_{fold}.pth"),
        )
        pd.DataFrame({"y_true": val_true, "y_pred": val_pred}).to_csv(
            os.path.join(args.result_dir, f"fold_{fold}_best_preds.csv"), index=False
        )

    print("\n=== Summary ===")
    print(f"PCC: {np.mean(fold_pccs):.4f} ± {np.std(fold_pccs):.4f}")
    print(f"MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"fold PCCs: {[round(x, 4) for x in fold_pccs]}")
    print("Reference (methods_datarisk Base_MLP, HKW): PCC = 0.231 ± 0.042")

    with open(os.path.join(args.result_dir, "cv_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"pcc_mean\t{np.mean(fold_pccs):.8g}\n")
        f.write(f"pcc_std\t{np.std(fold_pccs):.8g}\n")
        f.write(f"mae_mean\t{np.mean(fold_maes):.8g}\n")
        f.write(f"mae_std\t{np.std(fold_maes):.8g}\n")
        f.write(f"fold_pccs\t{fold_pccs}\n")
        f.write("input\tdosage\n")


if __name__ == "__main__":
    train_and_evaluate(parse_args())
