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

- `attention_weight_singleloci.py`  
  Extracts single-locus attention signals (from the `CNN_self_attention` branch) and exports averaged feature weights.

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

`data/data_zeamap/<phe>/`

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

- Data dir: `<project_root>/data/data_zeamap/<phe>/`
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

### Extract single-locus attention (single-loci)

```bash
python attention_weight_singleloci.py
```

#### What it does

- Loads a trained model checkpoint (default: fold 1)
- Runs a forward pass and extracts attention-related outputs from `CNN_self_attention`
- Computes averaged feature weights and exports to CSV

#### Default inputs/outputs (current script behavior)

**Important**: `attention_weight_singleloci.py` currently hard-codes the following in `main`:

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

1. **Paths**  
   - `Epiformer.py` uses project-relative paths (`data/` and `result/`) and supports overriding via `--data_dir/--result_dir`.
   - The two attention scripts still contain **hard-coded** paths/phenotype (see above). To make CLI args effective, remove those `args.phe = "KW/"` assignments.

2. **Argument overriding**  
   CLI args work for `Epiformer.py`. However, `attention_weight_singleloci.py` / `attention_weight_epi.py` override `argparse` values in `main`, so passing args from command line currently has no effect.

3. **CUDA / CPU**  
   The code automatically selects `cuda:0` if available; otherwise it falls back to CPU (slower).

4. **Reproducibility**  
   A fixed seed is set (`set_seed(42)`), but small variations may still occur across hardware / library versions.

---

### Citation

If you use Epiformer in your research, please cite:

> Zhang, X., Liu, L., Ren, L. et al. Epiformer: epistasis detection by genome language model and dual-channel network. *Genome Biology* (2026). https://doi.org/10.1186/s13059-026-04268-8

- Paper: https://link.springer.com/article/10.1186/s13059-026-04268-8
- DOI: https://doi.org/10.1186/s13059-026-04268-8

---

### Contact

- **Issues / Bugs**: please submit via repository Issues (include error logs, commands, environment info, and reproduction steps).
- **Email**: `<zhangxiaowei2118@gmail.com>`
- **Paper**: https://link.springer.com/article/10.1186/s13059-026-04268-8