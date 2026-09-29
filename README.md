# STRATA
Official implementation for the NeurIPS 2026 paper : Screening Lipid Nanoparticles through Structure-Ratio Alignment

## Environment Setup
We used uv as virtual environment
```console
foo@bar:~<position of working directory>/STRATA$ uv init --python 3.9
foo@bar:~<position of working directory>/STRATA$ uv run python          # Execute python (for initialization purpose), and then close it.
foo@bar:~<position of working directory>/STRATA$ uv pip install deepchem[torch]
foo@bar:~<position of working directory>/STRATA$ uv pip install torch_geometric tqdm pdbpp wandb
```

## How to train
Before training, read STRATA/dataset/README.md file, which includes instructions about datasets.

We specify the options that must be fixed in order to run the model, while the detailed hyperparameter search space is provided in the appendix of the paper.

For STRATA with RiPE, ROPE, APE and random positional embeddings, use train_model.py file.
```console
foo@bar:~<position of working directory>/STRATA/src$ uv run train_model.py --lr <learning rate> --dataset_name <dataset_name> --pos_enc <positional encodings> --consistency <alpha> --device <gpu number> --wd 0.01 --num_splits 10 --hidden <hidden dimension> --heads <number of heads>
```

For STRATA with GBF positional encodings, use train_model_comet_pos.py file.
```console
foo@bar:~<position of working directory>/STRATA/src$ uv run train_model_comet_pos.py --lr <learning rate> --dataset_name <dataset_name> --consistency <alpha> --device <gpu number> --wd 0.01 --num_splits 10 --hidden <hidden dimension> --heads <number of heads>
```

## Options for OOD splits.
For unseen molecule splits, use `--unseen_ILs`(A549 dataset), `--unseen_HLs` (other datasets).
For unseen ratio splits, use `--unseen_split_{IL/HL/CHOL/PEG/CAR}s`. 
Note that `--unseen_split_CARs` option creates unseen IL-to-cargo ratio split.
If none of such options are used, you are using random split.

## ETC
Note that `interaction_transformer` denotes STRATA model.

