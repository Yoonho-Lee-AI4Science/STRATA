# dataset preprocessing

## Environment Setup
Dataset preprocessing step requires its own environment.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv init --python 3.9
foo@bar:~<position of working directory>/STRATA/dataset$ uv run python          # Execute python (for initialization purpose), and then close it.
foo@bar:~<position of working directory>/STRATA/dataset$ uv pip install lmdb==1.4.0 ml-collections==0.1.1 numpy==1.23.4 scipy==1.9.3 tensorboardX==2.5.1 tqdm==4.64.1 tokenizers==0.13.2 pyprojroot==0.2.0 pandas==1.5.2 scikit-learn==1.2.0 rdkit-pypi==2022.9.3
```

## Preprocessed files
We uploaded preprocessed datasets as `preprocessed.tar.gz`.
To use preprocessed datasets,
```console
foo@bar:~<position of working directory>/STRATA/dataset$ tar -zxvf  preprocessed.tar.gz
foo@bar:~<position of working directory>/STRATA/dataset$ rm preprocessed.tar.gz
```

## How to preprocess datasets manually
The codes below are examples of preprocessing a549 dataset. 
Note that preprocess_selective_default.py preprocess LC2024 datasets.

For random split, run without additional options.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py 
```

For unseen molecule split, use `--unseen_ILs` for a549 dataset and `--unseen_HLs` for other datasets.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_ILs
```

For unseen IL ratio split, run with `--unseen_split_ILs` option.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_split_ILs
```

For unseen HL ratio split, run with `--unseen_split_HLs` option.
```console 
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_split_HLs
```

For unseen CHOL ratio split, run with `--unseen_split_CHOLs` option.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_split_CHOLs
```

For unseen PEG ratio split, run with `--unseen_split_PEGs` option.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_split_PEGs
```

For unseen IL-to-cargo ratio split,  run with `--unseen_split_CARs` option.
```console
foo@bar:~<position of working directory>/STRATA/dataset$ uv run ./preprocess_a549_default.py --unseen_split_CARs
```

After preprocessing, move the created datasets (located under `processed_data_dirs`) to `STRATA/dataset`.
For example, the datasets should be located as `STRATA/dataset/a549/A549_form_screen_usIL/0.6_0.2_0.2/...` format.

