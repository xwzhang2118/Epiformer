import os
import pickle
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--phe', type=str, default='KW', help='Dataset name (e.g., KW)')
    parser.add_argument('--runIndex', type=str, default='index1', help='Run index')
    parser.add_argument('--data_dir', type=str, default='', help='Path to phenotype directory (defaults to <project_root>/data/data_zeamap/<phe>/)')
    parser.add_argument('--result_dir', type=str, default='', help='Path to result directory (defaults to <project_root>/result/<phe>/)')
    parser.add_argument('--result_dir_base', type=str, default='', help='Base path to result directory')
    parser.add_argument('--result_directory_name', type=str, default='KW', help='Result directory name')
    parser.add_argument('--split', type=str, default='random', help='name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stop_patience', type=int, default=30, help='Early stopping patience')
    args = parser.parse_args()
    return args


def load_data(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    phe = args.phe.rstrip("/\\")
    if args.data_dir:
        args.data_dir = args.data_dir
    else:
        args.data_dir = os.path.join(project_root, "data", "data_zeamap", phe)
    if not args.data_dir.endswith(os.sep):
        args.data_dir += os.sep

    y_path_root = os.path.join(args.data_dir, "yData.csv")
    y_path_legacy = os.path.join(args.data_dir, "data_temporary", "yData.csv")
    y_path = y_path_root if os.path.exists(y_path_root) else y_path_legacy
    if phe == 'KW':
        yData = pd.read_csv(y_path, sep=',', header=None, index_col=0)
    else:
        yData = pd.read_csv(y_path, sep=',', header=None)

    xData_seq = np.load(args.data_dir + "genotype_encoded.npy")
    nsample, nsnp, dim = xData_seq.shape 

    SNP_annotation = pd.read_csv(args.data_dir  + "SNP_annotation.csv",  delimiter=',', index_col=0)
    SNP_annotation_tensor = torch.Tensor(SNP_annotation.to_numpy()).repeat(nsample,  1, 1)

    with open(args.data_dir + "embeddings.pkl", "rb") as f:
        xData_emb = pickle.load(f) 
    xData_emb = np.array(xData_emb)
    xData_emb = xData_emb.reshape(xData_emb.shape[0], xData_emb.shape[1], -1)
    xData = np.concatenate((xData_emb, xData_seq, SNP_annotation_tensor.numpy()), axis=2)
    
    xData = torch.Tensor(xData)
    feature_extractor3 = nn.Sequential(
                        nn.Linear(xData.shape[2], 256),
                        nn.ReLU(),
                        nn.Linear(256, 1),
                        nn.ReLU(),  
                    )
    xData = feature_extractor3(xData)
    xData = xData.reshape(nsample, -1)
    xData = xData.detach().numpy()
    scaler = StandardScaler()
    xData = scaler.fit_transform(xData)
    return xData, yData.values, nsample, nsnp

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[1, 3, 5, 7], stride=1, padding='same'):
        super(MultiScaleCNN, self).__init__()
        self.convs = nn.ModuleList()

        for k in kernel_sizes:
            pad = k // 2 if padding == 'same' else 0
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride, padding=pad)
            )
      
    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        out = torch.cat(features, dim=1) 
        out = torch.mean(out, dim=1, keepdim=True)
        return out

class CNN_self_attention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, attention_probs_dropout_prob=0.5):
        super(CNN_self_attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = MultiScaleCNN()

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.permute(0, 2, 1))
        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = mixed_query_layer
        key_layer = mixed_key_layer
        value_layer = mixed_value_layer

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.relu(hidden_states.view(hidden_states.size(0), -1))
        return hidden_states

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attn_weights  = self.attention(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x, attn_weights

class SNPInteractionAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, 
                 dropout, max_len, num_layers):
        super(SNPInteractionAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x_emb = self.embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x = x_emb
        for block in self.transformer_blocks:
            x, attn_weights = block(x)

        out = self.output(x)
        out = out.permute(1, 0, 2)
        return out

class CrossGatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossGatedMLP, self).__init__()
        self.mlp_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        self.gate_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_y = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x_proj = self.mlp_x(x)
        y_proj = self.mlp_y(y)

        x_gated = x_proj * self.gate_y(y)
        y_gated = y_proj * self.gate_x(x)

        out = torch.cat([x_gated, y_gated], dim=-1)
        return out

class Epiformer(nn.Module):
    def __init__(self,nsnp):
        super().__init__()
        self.nsnp = nsnp
        self.MCNN = CNN_self_attention(num_attention_heads=4, input_size=nsnp, hidden_size=512, output_dim=1, attention_probs_dropout_prob=0.1)
        self.SNP_interaction = SNPInteractionAttention(input_dim=1, embed_dim=512, num_heads=4, hidden_dim=64, dropout=0.1, max_len = 1000, num_layers=4)
        self.CrossGatedMLP = CrossGatedMLP(input_dim=self.nsnp, hidden_dim=self.nsnp)
        self.MLP = nn.Sequential(   
            nn.Linear(self.nsnp, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.linear = nn.Linear(self.nsnp*6, self.nsnp)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.nsnp, -1)
        x_M = self.MCNN(x)
        x_T = self.SNP_interaction(x).reshape(x.shape[0], -1)
        X_MT = x_M * x_T
        x1 = self.CrossGatedMLP(x_M, X_MT)
        x2 = self.CrossGatedMLP(x_T, X_MT)
        x3 = self.CrossGatedMLP(x_T, x_M)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = self.linear(x4)
        x_F = self.MLP(x5)

        return x_F 
    
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == '__main__':
  
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(42)

    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    phe = args.phe.rstrip("/\\")
    if not args.result_dir:
        args.result_dir = os.path.join(project_root, "result", phe)
    os.makedirs(args.result_dir, exist_ok=True)

    xData, yData, nsample, nsnp = load_data(args)
    ave_pcc = []
    ave_mae = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(xData)):
        print(f"Fold {fold + 1}")

        x_train, x_test = xData[train_idx], xData[test_idx]
        y_train, y_test = yData[train_idx], yData[test_idx]

        x_train = torch.Tensor(x_train).to(device)
        y_train = torch.Tensor(y_train).to(device)
        x_test = torch.Tensor(x_test).to(device)
        y_test = torch.Tensor(y_test).to(device)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        model = Epiformer(nsnp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        early_stopping = EarlyStopping(patience=args.early_stop_patience)
        best_loss = np.inf
        best_corr_coef = -float('inf')
        best_model_state = None
        early_stop_count = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            y_test_preds, y_test_trues = [], []
            att = []
            test_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    y_pred = model(x)
                    y_test_preds.extend(y_pred.cpu().numpy().reshape(-1).tolist())
                    y_test_trues.extend(y.cpu().numpy().reshape(-1).tolist())
                    test_loss += criterion(y_pred, y).item() * x.size(0)
            test_loss /= len(test_loader.dataset)
            corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]
            mae = mean_absolute_error(np.array(y_test_trues), np.array(y_test_preds))

            if corr_coef > best_corr_coef:
                result_df = pd.DataFrame({
                    'True': y_test_trues,
                    'Pred': y_test_preds
                })
                best_corr_coef = corr_coef
                best_mae = mae
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(args.result_dir, f'all_best_model_fold_{fold + 1}.pth'))

            early_stopping(corr_coef)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, PCC: {corr_coef:.4f}, MAE: {mae:.4f}")
        ave_pcc.append(best_corr_coef)
        ave_mae.append(best_mae)
        print(f"Best_PCC: {best_corr_coef:.4f}, Best_MAE: {best_mae:.4f}")

    print(f"Average PCC: {np.mean(ave_pcc):.4f}±{np.std(ave_pcc):.4f}, Average MAE: {np.mean(ave_mae):.4f}±{np.std(ave_mae):.4f}")
    print("successfully")