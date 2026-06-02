# CKG-LLMA: Confidence-aware KG-based Recommendation with LLM Augmentation

Reference implementation for the paper
**["Boosting Knowledge Graph-based Recommendations through Confidence-Aware Augmentation with Large Language Models"](https://arxiv.org/abs/2502.03715)**
(Rui Cai, Chao Wang, Qianyi Cai, Dazhong Shen, Hui Xiong).

CKG-LLMA enriches a recommendation knowledge graph (KG) with Large Language Models
while explicitly modelling **how much each piece of knowledge should be trusted**, so
that LLM hallucinations do not corrupt the recommendation signal.

---

## 1. What the method does

Knowledge-graph recommenders exploit semantic relations between users, items and
attributes, but real KGs are noisy, incomplete and expensive to curate. CKG-LLMA uses an
LLM to repair and extend the KG, and adds confidence modelling so unreliable triplets are
down-weighted instead of blindly trusted. It has four parts:

1. **LLM-based subgraph augmenter** — for each sampled user/item subgraph the LLM is asked
   to *add* missing item–attribute / item–item triplets and *delete* implausible ones,
   producing higher-quality KG views.
2. **Confidence-aware message propagation** — a relation-aware GAT propagates entity
   embeddings over the KG while learning per-edge confidence, so noisy or augmented
   triplets contribute less during aggregation.
3. **Dual-view contrastive learning** — contrastive objectives align the user–item
   interaction view with the KG view (and the original vs. LLM-augmented views), giving
   robust representations.
4. **Confidence-aware explanation generation** — reasoning paths through the KG, together
   with their learned confidence, are fed to an LLM to produce faithful, grounded
   recommendation explanations.

The recommendation backbone is LightGCN (BPR loss) combined with a TransE/TransR-style KG
embedding objective.

---

## 2. Repository layout

```
.
├── main.py                     # Main entry point: train / evaluate CKG-LLMA
├── kgcl.py                     # KGCL-style training entry point
├── subgraph.py                 # Subgraph extraction / explanation driver
├── gen_enhanced_graphs.py      # LLM-based KG augmentation (add/delete ui & ii triplets)
├── gen_item_attributes.py      # LLM-based item-attribute completion
├── Explanation_generation.py   # Interactive, path-based explanation generation
│
├── config/                     # YAML experiment configs
│   ├── CKG-LLMA/               #   main model configs + ablation/param sweeps
│   ├── KGCL/                   #   KGCL baseline configs
│   └── lightgcn/               #   LightGCN baseline configs
│
├── modules/
│   ├── data/                   # Datasets, samplers, KG/Rec loaders, data_config.py
│   ├── model/                  # CKG_LLMA, LightGCN, GAT, KG embeddings, MoE, Contrast
│   ├── procedure/              # train.py / test.py / procedure.py training loops
│   ├── utils/                  # config IO, losses, metrics, logging helpers
│   ├── prompts/                # LLM prompt templates (augmentation & explanation)
│   └── explanation/            # Explanation utilities
│
├── dataset/                    # Datasets (git-ignored; AmazonBook sample provided)
├── checkpoint/                 # Saved model checkpoints (git-ignored)
└── requirements.txt
```

---

## 3. Installation

```bash
# Python 3.10 is recommended (the compiled sampler targets cpython-310).
conda create -n ckg-llma python=3.10 -y
conda activate ckg-llma

# Install PyTorch 2.3.0 matching your CUDA version first, e.g.:
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# Then the remaining dependencies:
pip install -r requirements.txt
```

`requirements.txt` lists the `torch_geometric` / `pyg-lib` / `torch_scatter` /
`torch_sparse` / `torch_cluster` wheels as comments — install the variants matching your
CUDA/PyTorch build (e.g. the `+pt23cu121` wheels) from the
[PyG wheel index](https://data.pyg.org/whl/).

---

## 4. Credentials (LLM API)

The LLM-augmentation scripts call a chat-completion endpoint. **Keys are read from the
environment — never hard-code them.** Copy the template and fill it in:

```bash
cp .env.example .env
# edit .env, then export the variables into your shell, e.g.:
export LLM_API_KEY=...           # used by gen_enhanced_graphs.py / gen_item_attributes.py
export LLM_API_BASE=...          # optional; defaults to the HKUST-GZ gateway
export OPENAI_API_KEY=...         # used by the explanation module
```

---

## 5. Datasets

`modules/data/data_config.py` defines the supported datasets and their user/item counts:

| Dataset       | #Users  | #Items  |
|---------------|---------|---------|
| AmazonBook    | 13,373  | 37,837  |
| Steam         | 53,533  | 13,232  |
| Anime         | 18,394  | 10,228  |
| MovieLens100K | 943     | 1,675   |

A processed **AmazonBook** sample ships under `dataset/AmazonBook/` (interactions,
`kg.txt`, `entity2id.json`, `relation2id.json`, `id2name.json`, train/val/test splits, and
pre-saved tensors in `pre_saved/`). The `dataset/` directory is otherwise git-ignored;
place additional datasets there following the same structure.

---

## 6. End-to-end workflow

```
 (1) Extract subgraphs   →   (2) LLM augmentation   →   (3) Train CKG-LLMA   →   (4) Explain
     subgraph.py              gen_enhanced_graphs.py     main.py                 Explanation_generation.py
                              gen_item_attributes.py
```

### (1) Subgraph extraction (optional / precompute)
```bash
python subgraph.py --argpath config/CKG-LLMA/argsAB_origin.yaml
```

### (2) LLM-based KG augmentation
Generates per-batch `add`/`delete` triplet edits and item-attribute completions
(written to `saved_graphs/` and `dataset/<name>/LLM/`):
```bash
export LLM_API_KEY=...
python gen_enhanced_graphs.py --start 0 --end 8005
python gen_item_attributes.py
```

### (3) Train / evaluate the recommender
```bash
# Train (config controls dataset, model variant, contrastive/LLM switches, etc.)
python main.py --config config/CKG-LLMA/argsAB_origin.yaml

# Evaluate only: set `Train: False` in the config and provide `load_path`,
# or point at a config that already does so.
```
Configs for the other datasets and for the ablation / hyper-parameter studies live under
`config/CKG-LLMA/` (e.g. `argsST_origin.yaml`, `argsNM_origin.yaml`,
`AB_ablation/`, `AB_param/`, `ST_ablation/`, ...).

### (4) Confidence-aware explanation
```bash
export OPENAI_API_KEY=...
python Explanation_generation.py --config config/CKG-LLMA/argsAB_origin.yaml
# then type:  <user_id> <item_id>   (or "end" to quit)
```

---

## 7. Key configuration options

Configs are YAML (`config/CKG-LLMA/*.yaml`); a few important keys:

| Key                    | Meaning                                                       |
|------------------------|---------------------------------------------------------------|
| `data.name`            | Dataset (`AmazonBook`, `Steam`, ...)                          |
| `Train`                | `True` to train, `False` to evaluate only                    |
| `isContrastive`        | Enable dual-view contrastive learning                        |
| `isApplyLLMinfo`       | Use LLM-augmented triplets during training                   |
| `isConfiFilter`        | Enable confidence-aware filtering of triplets                |
| `ContrastiveSeperate` / `ContrastiveFused` | Contrastive-view construction mode        |
| `kgcn`                 | KG aggregator: `RGAT`, `GAT`, `MEAN`, `Ours`, `NO`           |
| `delete_ratio` / `add_ratio` | Fraction of LLM delete/add edits applied per step      |
| `embedding_dim`, `lightGCN_n_layers`, `learning_rate` | Backbone hyper-params       |
| `loss_con_weight`, `loss_reg_weight`, `nce_temperatue` | Loss weights / temperature |
| `topks`, `metrics`     | Evaluation cut-offs and metrics (precision / recall / ndcg)  |
| `save_path`, `load_path` | Checkpoint output / input paths                            |

> Adjust `cuda` (GPU index, or `-1` for CPU) and `save_path` to your environment before
> running — the shipped configs point at the authors' paths.

Experiment tracking uses [Weights & Biases](https://wandb.ai/) when `wandb: True`; run
`wandb login` first or set `wandb: False`.

---

## 8. Citation

```bibtex
@article{cai2025ckgllma,
  title   = {Boosting Knowledge Graph-based Recommendations through
             Confidence-Aware Augmentation with Large Language Models},
  author  = {Cai, Rui and Wang, Chao and Cai, Qianyi and Shen, Dazhong and Xiong, Hui},
  journal = {arXiv preprint arXiv:2502.03715},
  year    = {2025}
}
```
