# STRATA Datasets

Everything you need to get the data STRATA trains on, either ready-made or rebuilt from the raw CSV files.

[Overview](#overview) | [Option A: preprocessed](#option-a--use-the-preprocessed-splits-recommended) | [Option B: from raw CSV](#option-b--preprocess-from-the-raw-csv-files) | [Folder layout](#folder-layout) | [TSV format](#tsv-format) | [Back to main README](../README.md)

---

## Overview

STRATA learns from LNP formulations. Each sample has **four lipid components** (IL, HL, CHOL and PEG, given as SMILES), their **molar ratios**, an **IL-to-cargo ratio**, and a measured **transfection efficiency** that the model regresses.

| Dataset | `--dataset_name` | Raw file (`experiments/data_json/`) | Preprocessing script | What varies | # LNPs |
|---|---|---|---|---|---|
| A549 | `a549` | `lnp_ml.csv` | `preprocess_a549_default.py` | 192 ILs, 3 HLs, ratios | 1,801 |
| YZ22 | `yz22` | `lnpdb.csv` | `preprocess_yz22_default.py` | 1 IL, 6 HLs, ratios | 1,080 |
| LC24 – HEK293 | `LC24HEK293` | `LNP_selective.csv` | `preprocess_selective_default.py --cell HEK293` | 1 IL, 6 HLs, ratios | 1,080 |
| LC24 – N2a | `LC24N2a` | `LNP_selective.csv` | `preprocess_selective_default.py --cell N2a` | 1 IL, 4 HLs, ratios | 720 |
| LC24 – ARPE19 | `LC24ARPE19` | `LNP_selective.csv` | `preprocess_selective_default.py --cell ARPE19` | 1 IL, 4 HLs, ratios | 720 |
| LC24 – B16 | `LC24B16` | `LNP_selective.csv` | `preprocess_selective_default.py --cell B16` | 1 IL, 6 HLs, ratios | 1,080 |
| LC24 – PC3 | `LC24PC3` | `LNP_selective.csv` | `preprocess_selective_default.py --cell PC3` | 1 IL, 6 HLs, ratios | 1,080 |

> [!TIP]
> **Most users only need [Option A](#option-a--use-the-preprocessed-splits-recommended).** It gives you exactly the splits used in the paper, with no extra packages.

## Option A — Use the preprocessed splits (recommended)

We ship every dataset and split as `preprocessed.tar.gz` (about 785 MB, stored with Git LFS).

```bash
cd STRATA/dataset
tar -zxvf preprocessed.tar.gz
rm preprocessed.tar.gz            # optional: frees ~785 MB of disk space
```

After extraction, `dataset/` contains `a549/`, `yz22/`, `LC24HEK293/`, `LC24N2a/`, `LC24ARPE19/`, `LC24B16/` and `LC24PC3/`. You can start training right away (see the [main README](../README.md#quick-start)).

> [!NOTE]
> If `tar` says *"This does not look like a tar archive"*, you have the small Git LFS pointer file instead of the real archive. Run `git lfs install && git lfs pull` from the repository root, then try again.

## Option B — Preprocess from the raw CSV files

Use this option if you want to regenerate the splits, change the split ratios, or add another LC24 cell line.

### B-1. Environment setup (uv)

Preprocessing needs **its own uv setup inside `dataset/`**.

1. First finish Steps 0–2 of the [main installation guide](../README.md#installation-with-uv). The preprocessing scripts also import `torch`, which the main setup installs.
2. Then run these four commands in `STRATA/dataset`, exactly as written:

```bash
cd dataset             # from the repository root (STRATA/)
uv init --python 3.9
uv run python          # Execute python (for initialization purpose), and then close it.
uv pip install lmdb==1.4.0 ml-collections==0.1.1 numpy==1.23.4 scipy==1.9.3 tensorboardX==2.5.1 tqdm==4.64.1 tokenizers==0.13.2 pyprojroot==0.2.0 pandas==1.5.2 scikit-learn==1.2.0 rdkit-pypi==2022.9.3
```

> [!WARNING]
> Run `conda deactivate` first if a conda environment is active. Otherwise `uv pip install` installs into the conda environment instead of the uv environment.

### B-2. Run a preprocessing script

Run the scripts from inside `dataset/`. Each run creates **one split type** for **one dataset**, with 20 random seeds (`split_0` … `split_19`).

```bash
cd STRATA/dataset

# A549 ─ random split (no flag)
uv run preprocess_a549_default.py

# YZ22 ─ random split
uv run preprocess_yz22_default.py

# LC24 ─ choose the cell line with --cell (N2a, HEK293, ARPE19, B16, PC3)
uv run preprocess_selective_default.py --cell HEK293
```

To build an out-of-distribution split, add **one** flag:

```bash
uv run preprocess_a549_default.py --unseen_ILs                       # unseen ILs       (A549 only)
uv run preprocess_yz22_default.py --unseen_HLs                       # unseen HLs       (YZ22, LC24)
uv run preprocess_selective_default.py --cell N2a --unseen_split_PEGs  # unseen PEG ratios
```

| Split | Flag | Output folder suffix | Available for |
|---|---|---|---|
| Random | *(none)* | — | all |
| Unseen IL molecules | `--unseen_ILs` | `_usIL` | A549 |
| Unseen HL molecules | `--unseen_HLs` | `_usHL` | YZ22, LC24 |
| Unseen IL ratio | `--unseen_split_ILs` | `_usILratio` | all |
| Unseen HL ratio | `--unseen_split_HLs` | `_usHLratio` | all |
| Unseen CHOL ratio | `--unseen_split_CHOLs` | `_usCHOLratio` | all |
| Unseen PEG ratio | `--unseen_split_PEGs` | `_usPEGratio` | all |
| Unseen IL-to-cargo ratio | `--unseen_split_CARs` | `_usCARratio` | all |

<details>
<summary><b>Want every split for a dataset? Use a loop</b></summary>

```bash
cd STRATA/dataset
for flag in "" --unseen_ILs --unseen_split_ILs --unseen_split_HLs \
            --unseen_split_CHOLs --unseen_split_PEGs --unseen_split_CARs; do
    uv run preprocess_a549_default.py $flag
done
```

For YZ22 or LC24, replace `--unseen_ILs` with `--unseen_HLs` and change the script name (add `--cell <CELL>` for LC24).

</details>

<details>
<summary><b>Split sizes</b></summary>

- **A549**: 60 / 20 / 20 % (train / valid / test). With `--unseen_split_CARs`, the split is 400 / 400 / 400.
- **YZ22, LC24 – HEK293 / B16 / PC3**: 540 / 180 / 180. With `--unseen_split_CARs`, the split is 360 / 360 / 360.
- **LC24 – N2a / ARPE19**: 360 / 120 / 120. With `--unseen_split_CARs`, the split is 240 / 240 / 240.

The IL-to-cargo ratio takes only a few distinct values, so that split is 1 : 1 : 1. Within each dataset, the other split types use (nearly) the same set sizes, which keeps the results comparable.

</details>

### B-3. Move the results into `dataset/`

The scripts write to `STRATA/experiments/processed_data_dirs/`. Training expects the data under `STRATA/dataset/<dataset_name>/`, so move the folders there:

```bash
cd STRATA

# A549
mkdir -p dataset/a549
mv experiments/processed_data_dirs/A549_form_screen* dataset/a549/

# YZ22
mkdir -p dataset/yz22
mv experiments/processed_data_dirs/YZ22_form_screen* dataset/yz22/

# LC24 (repeat for each cell line you built)
mkdir -p dataset/LC24HEK293
mv experiments/processed_data_dirs/LC24HEK293_form_screen* dataset/LC24HEK293/
```

The final path must look like `STRATA/dataset/a549/A549_form_screen_usIL/0.6_0.2_0.2/split_0/...`.

## Folder layout

```
dataset/
├── a549/
│   ├── A549_form_screen/                  # random split
│   │   ├── mol.lmdb
│   │   └── 0.6_0.2_0.2/
│   │       ├── split_0/
│   │       │   ├── mol.lmdb
│   │       │   └── A549_form_screen/
│   │       │       ├── train.tsv  valid.tsv  test.tsv     # <- read by STRATA
│   │       │       └── train.lmdb valid.lmdb test.lmdb    # COMET format, not used by STRATA
│   │       ├── split_1/
│   │       └── ... split_19/
│   ├── A549_form_screen_usIL/             # unseen-IL split
│   ├── A549_form_screen_usILratio/        # unseen-IL-ratio split
│   └── ...
├── yz22/        └── YZ22_form_screen*/
└── LC24N2a/     └── LC24N2a_form_screen*/   (same for LC24HEK293, LC24ARPE19, LC24B16, LC24PC3)
```

How the training flags map to a folder:

```
dataset/<dataset_name>/<FORM>_form_screen<suffix>/0.6_0.2_0.2/split_<i>/<FORM>_form_screen/{train,valid,test}.tsv
        └ --dataset_name  └ A549 / YZ22 / LC24<CELL>  └ split flag              └ 0 … num_splits-1
```

STRATA itself reads **only the `.tsv` files**. The `.lmdb` files (and the `.json` debug copies that the scripts also write) are kept for compatibility with the original COMET pipeline.

## TSV format

Each `train/valid/test.tsv` file is tab-separated with one LNP formulation per row:

| Column | Meaning |
|---|---|
| `lnp_id` | Formulation ID (written to `result.txt` with the predictions) |
| `label` | Transfection efficiency, the regression target |
| `IL_SMILES`, `HL_SMILES`, `CHOL_SMILES`, `PEG_SMILES` | SMILES of the four lipid components |
| `HL_name` | Helper-lipid name |
| `mRNA_weight` | IL-to-cargo ratio (e.g. IL : mRNA weight ratio in A549, N/P ratio in LC24) |
| `IL_ratio`, `HL_ratio`, `CHOL_ratio`, `PEG_ratio` | Molar ratios (%) of the four components |
| `dataset_name` | Folder tag, e.g. `A549_form_screen` |

> [!TIP]
> **Using your own data?** Write TSV files with these columns into the same folder structure, for example `dataset/mydata/mydata_form_screen/0.6_0.2_0.2/split_0/mydata_form_screen/train.tsv`. Then train with `--dataset_name mydata`.

---

**Ready-made splits in one `tar` command, fully reproducible splits in one `uv run`.**
