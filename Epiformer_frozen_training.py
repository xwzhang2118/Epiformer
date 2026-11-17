"""Public Epiformer.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# HKW keep trial11 backbone (source of the 0.461±0.082 run)
KEEP_MODEL_PARAMS = {
    "mcnn_heads": 1,
    "mcnn_hidden": 8,
    "mcnn_dropout": 0.2302079928940613,
    "trans_embed_dim": 32,
    "trans_heads": 1,
    "trans_hidden_dim": 64,
    "trans_dropout": 0.2453478870137169,
    "trans_layers": 6,
    "cross_hidden_dim": 8,
    "mlp_hidden1": 128,
    "mlp_hidden2": 32,
}
KEEP_LR = 0.002406375651714279
KEEP_BATCH_SIZE = 16
KEEP_EPOCHS = 60


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Epiformer (frozen trained fusion, no final ReLU)")
    p.add_argument("--phe", type=str, default="HKW", help="Dataset name (e.g., HKW or KW)")
    p.add_argument("--data_dir", type=str, default="", help="Data directory")
    p.add_argument("--result_dir", type=str, default="", help="Result directory")
    p.add_argument("--batch_size", type=int, default=KEEP_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=KEEP_EPOCHS)
    p.add_argument("--lr", type=float, default=KEEP_LR)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=int, default=1, help="Fusion output dim")
    p.add_argument(
        "--fusion",
        type=str,
        default="frozen_trained_mlp_no_relu",
        choices=["frozen_trained_mlp_no_relu", "frozen_trained_mlp"],
        help="Default matches the reported 0.461±0.082 protocol",
    )
    return p.parse_args()


# -------------------------------
# Data
# -------------------------------
def load_raw_features(args):
    """Load multimodal features as (N, P, F) without fusion."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    phe = args.phe.rstrip("/\\")
    if getattr(args, "data_dir", None):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(project_root, "data", "data_zeamap", phe)
    if not data_dir.endswith(os.sep):
        data_dir += os.sep
    args.data_dir = data_dir

    y_path_root = os.path.join(data_dir, "yData.csv")
    y_path_legacy = os.path.join(data_dir, "data_temporary", "yData.csv")
    y_path = y_path_root if os.path.exists(y_path_root) else y_path_legacy
    # Prefer numeric-only phenotype; fall back to legacy KW (sample-id index).
    y_df = pd.read_csv(y_path, sep=",", header=None)
    if y_df.shape[1] >= 2 and phe == "KW":
        # legacy KW: col0=sample id, col1=phenotype
        yData = np.asarray(y_df.iloc[:, 1].values, dtype=np.float64).reshape(-1)
    else:
        yData = np.asarray(y_df.values, dtype=np.float64).reshape(-1)

    x_seq = np.load(os.path.join(data_dir, "genotype_encoded.npy")).astype(np.float32)
    nsample, nsnp, _ = x_seq.shape

    ann_path = os.path.join(data_dir, "SNP_annotation.csv")
    # Prefer numeric-only annotation (no SNP-name index). Fall back to legacy index_col=0.
    ann = pd.read_csv(ann_path, delimiter=",", header=None)
    if ann.shape[0] != nsnp:
        ann = pd.read_csv(ann_path, delimiter=",", index_col=0)
    if ann.shape[0] != nsnp:
        raise ValueError(f"SNP_annotation rows {ann.shape[0]} != nsnp {nsnp}")
    ann_np = np.asarray(ann.to_numpy(), dtype=np.float32)
    x_ann = np.broadcast_to(ann_np[None, :, :], (nsample, nsnp, ann_np.shape[1])).copy()

    with open(os.path.join(data_dir, "embeddings.pkl"), "rb") as f:
        x_emb = np.asarray(pickle.load(f), dtype=np.float32)
    x_emb = x_emb.reshape(x_emb.shape[0], x_emb.shape[1], -1)

    xData = np.concatenate((x_emb, x_seq, x_ann), axis=2).astype(np.float32)
    nfeat = xData.shape[2]
    print(f"[{phe}] raw X={xData.shape} (N, nsnp, F)")
    return xData, yData, nsample, nsnp, nfeat


def scale_features_fold(x_train: np.ndarray, x_apply: np.ndarray):
    """Per-feature StandardScaler fit on train only. Shapes: (N, P, F)."""
    n_tr, p, f = x_train.shape
    n_ap = x_apply.shape[0]
    sc = StandardScaler()
    x_tr = sc.fit_transform(x_train.reshape(-1, f)).reshape(n_tr, p, f).astype(np.float32)
    x_ap = sc.transform(x_apply.reshape(-1, f)).reshape(n_ap, p, f).astype(np.float32)
    return x_tr, x_ap, sc


# -------------------------------
# Fusion
# -------------------------------
class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, final_relu: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)]
        if final_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _apply_frozen_mlp(mlp: nn.Module, X: np.ndarray, device: torch.device, bs: int = 8) -> np.ndarray:
    mlp.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=device)
            outs.append(mlp(xb).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def pretrain_then_freeze_fusion(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    C: int,
    final_relu: bool,
    device: torch.device,
    seed: int,
    hidden: int = 256,
    lr: float = 1e-3,
    max_epochs: int = 150,
    patience: int = 25,
    batch_size: int = 8,
    weight_decay: float = 1e-4,
    readout: str = "flat",
) -> tuple[np.ndarray, np.ndarray, FusionMLP, StandardScaler, dict]:
    """Fold-wise: F-scale → supervise-pretrain FusionMLP → freeze → fuse train/val."""
    n_tr, nsnp, F = X_train.shape
    n_va = X_val.shape[0]
    X_tr, X_va, sc = scale_features_fold(X_train, X_val)

    idx = np.arange(n_tr)
    tr_i, es_i = train_test_split(idx, test_size=0.1, random_state=seed)
    y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_mu, y_sd = float(y_tr[tr_i].mean()), float(max(y_tr[tr_i].std(), 1e-8))
    y_tr_z = ((y_tr[tr_i] - y_mu) / y_sd).astype(np.float32)
    y_es_z = ((y_tr[es_i] - y_mu) / y_sd).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    mlp = FusionMLP(F, C, hidden=hidden, final_relu=final_relu).to(device)
    if readout == "flat":
        head = nn.Linear(nsnp * C, 1).to(device)
    elif readout == "mean":
        head = nn.Linear(C, 1).to(device)
    else:
        raise ValueError(readout)
    opt = optim.Adam(
        list(mlp.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay
    )

    Xtr_t = torch.tensor(X_tr[tr_i], device=device)
    Yes_t = torch.tensor(X_tr[es_i], device=device)
    ytr_t = torch.tensor(y_tr_z, device=device)

    def _pred(xb: torch.Tensor) -> torch.Tensor:
        z = mlp(xb)
        if readout == "flat":
            return head(z.reshape(z.shape[0], -1)).squeeze(-1)
        return head(z.mean(dim=1)).squeeze(-1)

    best_score, best_state, stall = -1e9, None, 0
    for _ep in range(1, max_epochs + 1):
        mlp.train()
        head.train()
        perm = torch.randperm(len(tr_i), device=device)
        for i in range(0, len(tr_i), batch_size):
            bi = perm[i : i + batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(_pred(Xtr_t[bi]), ytr_t[bi])
            loss.backward()
            opt.step()
        mlp.eval()
        head.eval()
        with torch.no_grad():
            pred_es = _pred(Yes_t).cpu().numpy()
        score = _regression_metrics(y_es_z, pred_es)[3]
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    for p in mlp.parameters():
        p.requires_grad_(False)
    mlp.eval()

    X_tr_f = _apply_frozen_mlp(mlp, X_tr, device)
    X_va_f = _apply_frozen_mlp(mlp, X_va, device)
    meta = {
        "fusion_pretrain_val_pcc": float(best_score if np.isfinite(best_score) else 0.0),
        "fusion_final_relu": bool(final_relu),
        "fusion_hidden": int(hidden),
        "fusion_C": int(C),
        "fusion_readout": readout,
    }
    return X_tr_f, X_va_f, mlp, sc, meta


# -------------------------------
# Backbone
# -------------------------------
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=None, stride=1, padding="same"):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=(k // 2 if padding == "same" else 0),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return torch.mean(out, dim=1, keepdim=True)


class CNN_self_attention(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        input_size,
        hidden_size,
        output_dim=1,
        attention_probs_dropout_prob=0.5,
        in_channels: int = 1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = nn.LayerNorm(input_size, eps=1e-12)
        self.relu = nn.ReLU()
        self.cnn = MultiScaleCNN(in_channels=in_channels, out_channels=1)

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.permute(0, 2, 1))
        input_tensor = cnn_hidden
        q = self.query(input_tensor)
        k = self.key(input_tensor)
        v = self.value(input_tensor)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.attention_head_size)
        probs = self.attn_dropout(torch.nn.Softmax(dim=-1)(scores))
        context = torch.matmul(probs, v)
        hidden = self.LayerNorm(self.dense(context) + input_tensor)
        return self.relu(hidden.view(hidden.size(0), -1))


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, need_weights: bool = False):
        attn_output, attn_w = self.attention(x, x, x, need_weights=need_weights)
        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x, attn_w


class SNPInteractionAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, dropout, max_len, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, x, return_attn: bool = False):
        x = self.pos_encoding(self.embedding(x))
        last_attn = None
        for i, block in enumerate(self.transformer_blocks):
            need = return_attn and (i == len(self.transformer_blocks) - 1)
            x, attn_w = block(x, need_weights=need)
            if need:
                last_attn = attn_w
        out = self.output(x)
        if return_attn:
            return out, last_attn
        return out


class CrossGatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp_x = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.mlp_y = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.gate_x = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.gate_y = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

    def forward(self, x, y):
        return torch.cat(
            [self.mlp_x(x) * self.gate_y(y), self.mlp_y(y) * self.gate_x(x)], dim=-1
        )


class Epiformer(nn.Module):
    """Epiformer backbone. Optional frozen ``fusion`` for inference / attention export."""

    def __init__(
        self,
        nsnp,
        feature_dim: int = 1,
        in_features: int | None = None,
        fusion_hidden: int = 256,
        fusion_out: int = 1,
        fusion_final_relu: bool = False,
        attach_fusion: bool = False,
        mcnn_heads=1,
        mcnn_hidden=8,
        mcnn_dropout=0.2302079928940613,
        trans_embed_dim=32,
        trans_heads=1,
        trans_hidden_dim=64,
        trans_dropout=0.2453478870137169,
        trans_layers=6,
        cross_hidden_dim=8,
        mlp_hidden1=128,
        mlp_hidden2=32,
    ):
        super().__init__()
        self.nsnp = nsnp
        self.feature_dim = feature_dim
        self.in_features = in_features
        self.fusion_hidden = fusion_hidden
        self.fusion_out = fusion_out
        self.fusion = None
        if attach_fusion:
            if in_features is None:
                raise ValueError("attach_fusion requires in_features")
            self.fusion = FusionMLP(
                in_features, fusion_out, hidden=fusion_hidden, final_relu=fusion_final_relu
            )

        self.MCNN = CNN_self_attention(
            num_attention_heads=mcnn_heads,
            input_size=nsnp,
            hidden_size=mcnn_hidden,
            output_dim=1,
            attention_probs_dropout_prob=mcnn_dropout,
            in_channels=feature_dim,
        )
        self.SNP_interaction = SNPInteractionAttention(
            input_dim=feature_dim,
            embed_dim=trans_embed_dim,
            num_heads=trans_heads,
            hidden_dim=trans_hidden_dim,
            dropout=trans_dropout,
            max_len=nsnp,
            num_layers=trans_layers,
        )
        self.CrossGatedMLP = CrossGatedMLP(input_dim=self.nsnp, hidden_dim=cross_hidden_dim)
        self.MLP = nn.Sequential(
            nn.Linear(cross_hidden_dim * 6, mlp_hidden1),
            nn.ReLU(),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Linear(mlp_hidden2, 1),
        )

    def forward(self, x, return_attn: bool = False):
        if x.dim() == 2:
            x = x.view(x.shape[0], self.nsnp, -1)
        if self.fusion is not None and x.shape[-1] == self.in_features:
            x = self.fusion(x)
        if return_attn:
            x_T, attn = self.SNP_interaction(x, return_attn=True)
            x_T = x_T.reshape(x.shape[0], -1)
        else:
            x_T = self.SNP_interaction(x).reshape(x.shape[0], -1)
            attn = None
        x_M = self.MCNN(x)
        X_MT = x_M * x_T
        x1 = self.CrossGatedMLP(x_M, X_MT)
        x2 = self.CrossGatedMLP(x_T, X_MT)
        x3 = self.CrossGatedMLP(x_T, x_M)
        y = self.MLP(torch.cat([x1, x2, x3], dim=1))
        if return_attn:
            return y, attn
        return y


class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
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

    xData, yData, nsample, nsnp, nfeat = load_raw_features(args)
    if not args.result_dir:
        args.result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "result", args.phe.rstrip("/\\")
        )
    os.makedirs(args.result_dir, exist_ok=True)

    final_relu = args.fusion == "frozen_trained_mlp"
    C = int(args.C)
    model_params = dict(KEEP_MODEL_PARAMS)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_pccs, fold_maes = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(xData), 1):
        print(f"\n=== Fold {fold} | fusion={args.fusion} C={C} ===")
        fold_seed = args.seed + fold
        set_seed(fold_seed)

        X_tr_f, X_va_f, fusion_mlp, feat_scaler, fus_meta = pretrain_then_freeze_fusion(
            xData[train_idx],
            yData[train_idx],
            xData[val_idx],
            C=C,
            final_relu=final_relu,
            device=device,
            seed=args.seed + 7 + fold,
        )
        print(
            f"  [frozen fusion] pretrain_val_pcc={fus_meta['fusion_pretrain_val_pcc']:.4f} "
            f"final_relu={fus_meta['fusion_final_relu']} readout={fus_meta['fusion_readout']}",
            flush=True,
        )

        sc2 = StandardScaler()
        X_train_np = sc2.fit_transform(X_tr_f.reshape(X_tr_f.shape[0], -1)).astype(np.float32)
        X_val_np = sc2.transform(X_va_f.reshape(X_va_f.shape[0], -1)).astype(np.float32)

        model = Epiformer(nsnp, feature_dim=C, attach_fusion=False, **model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=args.early_stop_patience)

        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
        y_train = torch.tensor(yData[train_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(yData[val_idx], dtype=torch.float32, device=device)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

        best_pcc = -float("inf")
        best_sd = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(X_train.size(0), device=device)
            last_loss = 0.0
            for i in range(0, X_train.size(0), args.batch_size):
                bi = perm[i : i + args.batch_size]
                optimizer.zero_grad()
                pred = model(X_train[bi])
                loss = criterion(pred, y_train[bi].view(-1, 1))
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

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
                f"Epoch {epoch}, Train Loss: {last_loss:.4f} | "
                f"Val PCC: {pcc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}"
            )
            early_stopping(pcc)
            if pcc > best_pcc:
                best_pcc = pcc
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if early_stopping.early_stop:
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

        # Inference-friendly checkpoint: backbone + frozen fusion
        infer_model = Epiformer(
            nsnp,
            feature_dim=C,
            in_features=nfeat,
            fusion_hidden=256,
            fusion_out=C,
            fusion_final_relu=final_relu,
            attach_fusion=True,
            **model_params,
        )
        infer_model.load_state_dict(model.state_dict(), strict=False)
        infer_model.fusion.load_state_dict(fusion_mlp.state_dict())
        for p in infer_model.fusion.parameters():
            p.requires_grad_(False)

        ckpt = {
            "model_state_dict": infer_model.state_dict(),
            "backbone_state_dict": best_sd,
            "fusion_state_dict": fusion_mlp.state_dict(),
            "nsnp": nsnp,
            "in_features": nfeat,
            "fusion_hidden": 256,
            "fusion_out": C,
            "fusion_final_relu": final_relu,
            "fusion": args.fusion,
            "C": C,
            "fold": fold,
            "pcc": float(pcc_f),
            "feat_scaler_mean": feat_scaler.mean_.tolist(),
            "feat_scaler_scale": feat_scaler.scale_.tolist(),
            "post_fusion_scaler_mean": sc2.mean_.tolist(),
            "post_fusion_scaler_scale": sc2.scale_.tolist(),
            "protocol": "frozen_trained_mlp_no_relu" if not final_relu else "frozen_trained_mlp",
            "reported_hkw_keep_pcc": "0.461±0.082",
        }
        torch.save(ckpt, os.path.join(args.result_dir, f"all_best_model_fold_{fold}.pth"))
        pd.DataFrame({"y_true": val_true, "y_pred": val_pred}).to_csv(
            os.path.join(args.result_dir, f"fold_{fold}_best_preds.csv"), index=False
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(f"PCC: {np.mean(fold_pccs):.4f} ± {np.std(fold_pccs):.4f}")
    print(f"MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"fold PCCs: {[round(x, 4) for x in fold_pccs]}")
    print(
        "Note: HKW keep reference for this protocol is PCC = 0.461 ± 0.082 "
        "(seed=42; demo KW may differ)."
    )


if __name__ == "__main__":
    train_and_evaluate(parse_args())
