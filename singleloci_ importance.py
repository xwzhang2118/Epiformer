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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--runIndex', type=str, default='index1', help='Run index')
    parser.add_argument('--data_dir', type=str, default='', help='Path to data directory')
    parser.add_argument('--result_dir', type=str, default='', help='Path to result directory')
    parser.add_argument('--result_dir_base', type=str, default='', help='Base path to result directory')
    parser.add_argument('--result_directory_name', type=str, default='MVP_DTA', help='Result directory name')
    parser.add_argument('--split', type=str, default='random', help='name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stop_patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='result/model/epoch_model', help='path')
    args = parser.parse_args()
    return args

def load_data(args):
    args.data_dir = os.path.join("demo_data/data_zeamap/", args.phe)

    if args.phe=='KW/':
        yData = pd.read_csv(args.data_dir + "/yData.csv", sep=',', header=None, index_col=0)
    else:
        yData = pd.read_csv(args.data_dir + "/yData.csv", sep=',', header=None)

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
                        nn.Linear(373, 256),
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

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)

        self.head_importance = nn.Parameter(torch.ones(num_attention_heads))

        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = MultiScaleCNN()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.permute(0, 2, 1))
        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        head_importance_normalized = self.head_importance / self.head_importance.sum()
        head_weighted_output = context_layer * head_importance_normalized.unsqueeze(-1).unsqueeze(-1)
        self.update_head_importance(attention_probs)
        context_layer = head_weighted_output.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_size)


        hidden_states1 = self.dense(context_layer)
        hidden_states = self.LayerNorm(hidden_states1 + input_tensor)
        hidden_states = self.relu(hidden_states.view(hidden_states.size(0), -1))


        return hidden_states, hidden_states1
    
    def update_head_importance(self, attention_probs):

        self.head_importance.data += attention_probs.mean(dim=(0, 1)).sum(dim=-1)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding[:x.size(0), :, :]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
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
        self.SNP_interaction = SNPInteractionAttention(input_dim=1, embed_dim=512, num_heads=4, hidden_dim=64, dropout=0.1, max_len=nsnp, num_layers=4)
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
        x_M, att = self.MCNN(x)
        return x_M, att
    
    
    
def load_model_and_extract_weights(model_path, nsnp):

    model = Epiformer(nsnp).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print(model)
    model.eval()

    return model

def extract_attention_weights(model, x_test_tensor):
    model.eval()
    with torch.no_grad():
        _, attention_probs = model(x_test_tensor)

    print(f"Attention weights shape: {attention_probs.shape}")

    attention_weights = attention_probs.cpu().numpy()
    return attention_weights

def attention_weights_to_dataframe(attention_weights):
    print(f"Attention weights shape: {attention_weights.shape}")

    num_samples, num_heads, seq_len = attention_weights.shape
    attention_list = []

    for sample_idx in range(num_samples):
        for head_idx in range(num_heads):
            for seq_idx in range(seq_len):
                row = {
                    "Sample": sample_idx,
                    "Head": head_idx,
                    "Position": seq_idx,
                    "Weight": attention_weights[sample_idx, head_idx, seq_idx].mean()
                }
                attention_list.append(row)

    attention_df = pd.DataFrame(attention_list)
    return attention_df

def attention_weights_to_avg_feature_weights(attention_weights):
    print(f"Attention weights shape: {attention_weights.shape}")

    avg_feature_weights = attention_weights.mean(axis=(0, 1))
    feature_weight_df = pd.DataFrame({
        "Feature": np.arange(len(avg_feature_weights)),
        "AverageAttentionWeight": avg_feature_weights
    })
    return feature_weight_df

if __name__ == '__main__':
  
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(42)

    args = parse_args()  
    args.phe = "KW/"
    args.result_dir = "result/KW"
    xData, yData, nsample, nsnp = load_data(args)

    model_path = 'result/KW/all_best_model_fold_1.pth'
 
    data = xData
    data = torch.from_numpy(data).float().to(device)

    model = load_model_and_extract_weights(model_path, nsnp)

    attention_weights = extract_attention_weights(model, data)

    attention_df = attention_weights_to_dataframe(attention_weights)

    avg_feature_weights_df = attention_weights_to_avg_feature_weights(attention_weights)

    avg_feature_weights_df.to_csv("result/KW/attention_weights_test_3.csv", index=False)

    print("successfully")
