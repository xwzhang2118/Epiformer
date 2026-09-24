# Epiformer

## English

Epiformer is a PyTorch-based deep learning framework for **epistasis detection** and **phenotype prediction**. It builds an end-to-end regression model (`Epiformer`) by combining:

- Multi-scale CNN + self-attention (`CNN_self_attention`)
- SNP interaction Transformer (`SNPInteractionAttention`)
- Cross-Gated MLP fusion (`CrossGatedMLP`)

### Framework

![Epiformer framework](result/KW/Epiformer.png)

---

### Project files

- `Epiformer.py`  
  Main training script. Runs 5-fold cross-validation, selects and saves the best model by PCC (Pearson correlation coefficient).

- `singleloci_ importance.py`  
  Extracts single-locus (per-SNP) importance scores (from the `CNN_self_attention` branch) and exports averaged feature weights.

- `attention_weight_epi.py`  
  Extracts SNP–SNP interaction attention matrices (from the Transformer branch) and exports an averaged attention matrix.

---

### Environment setup

We recommend using conda (the project root provides `environment.yml`):

```bash
# In the project root
conda env create -f environment.yml
conda activate Epiformer
```

Minimal dependencies (for reference): Python 3.9+ and

```bash
pip install torch numpy pandas scikit-learn
```

---

### Data preparation

By default, scripts read data from the following directory (relative to the project root):

`demo_data/data_zeamap/<phe>/`

where `<phe>` is the phenotype name (e.g., `KW`). The directory should contain at least:

- `genotype_encoded.npy`
- `embeddings.pkl`
- `SNP_annotation.csv`
- `yData.csv`

> Dataset download link / DOI: (https://doi.org/10.6084/m9.figshare.30448166)

---

### Outputs (default)

- Training and interpretation outputs are written to:
  - `result/<phe>/`

---

### Train the model

```bash
python Epiformer.py --phe KW
```

`Epiformer.py` supports CLI arguments. If `--data_dir/--result_dir` are not provided, it uses:

- Data dir: `<project_root>/demo_data/data_zeamap/<phe>/`
- Result dir: `<project_root>/result/<phe>/`

#### Training procedure (`Epiformer.py`)

1. Load and concatenate multi-source features (embedding + genotype + SNP annotation).
2. Reduce feature dimension with a small MLP and apply standardization.
3. Train `Epiformer` with 5-fold cross-validation.
4. For each epoch, compute:
   - `Train Loss` (MSE)
   - `Test Loss` (MSE)
   - `PCC`
   - `MAE`
5. If PCC improves for a fold, save the best model:
   - `all_best_model_fold_<k>.pth`
6. Use Early Stopping (default patience=30) based on PCC.

#### Results

- Prints best `PCC / MAE` for each fold
- Prints the final 5-fold average: `Average PCC ± std` and `Average MAE ± std`
- Saves best models (per fold) to:
  - `result/<phe>/all_best_model_fold_<k>.pth`

---

### Extract single-locus importance 

```bash
python "singleloci_ importance.py"
```

#### What it does

- Loads a trained model checkpoint (default: fold 1)
- Extracts single-locus (per-SNP) importance scores from the local `CNN_self_attention` branch
- Averages scores across samples and exports one importance value per SNP to CSV

#### Default inputs/outputs (current script behavior)

**Important**: `singleloci_ importance.py` currently hard-codes the following in `main`:

- `args.phe = "KW/"`
- `args.result_dir = "result/KW"`
- `model_path = "result/KW/all_best_model_fold_1.pth"`

Default output file:

- `result/KW/attention_weights_test_3.csv`

---

### Extract SNP interaction attention (epistasis)

```bash
python attention_weight_epi.py
```

#### What it does

- Loads a trained model checkpoint
- Extracts the attention matrix from the `SNPInteractionAttention` branch
- Averages over samples and exports an SNP×SNP attention matrix

#### Default inputs/outputs (current script behavior)

**Important**: `attention_weight_epi.py` currently hard-codes the following in `main`:

- `args.phe = "KW/"`
- `args.result_dir = "result/"`
- `model_path = "result/KW/all_best_model_fold_1.pth"`

Default output file:

- `result/KW/attention_weights_matrix_test.csv`

---

### FAQ / Notes

1. **CUDA / CPU**  
   The code automatically selects `cuda:0` if available; otherwise it falls back to CPU (slower).

2. **Reproducibility**  
   A fixed seed is set (`set_seed(42)`), but small variations may still occur across hardware / library versions.

3. **SNP contextual embeddings (Evo 2)**  
   Due to differences across experimental environments, we recommend that users use **Evo 2** to re-embed the contextual sequence around each SNP (typically 10 bp upstream and 10 bp downstream, plus the SNP itself), then save the resulting embeddings as `embeddings.pkl` under `demo_data/data_zeamap/<phe>/` before training or inference.

---

### Contact
- **Email**: `<zhangxiaowei2118@gmail.com>`