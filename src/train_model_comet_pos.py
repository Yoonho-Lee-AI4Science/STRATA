import argparse
import os
import pdb
from train_comet_pos import run_training


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DATASET_DIRS = {
    "a549": "a549",
    "xh25": "xh25",
    "yz22": "yz22",
    "yz24": "yz24",
}
LEGACY_FORM_PREFIXES = {
    "a549": "A549",
    "xh25": "XH25",
    "yz22": "YZ22",
    "yz24": "YZ24",
}


def _device_string(device_arg):
    """Normalize device input into a cuda device string."""
    if str(device_arg).startswith("cuda"):
        return str(device_arg)
    return f"cuda:{device_arg}"


def _get_config_string(args):
    """Build a compact config string for experiment folders."""
    ignore_keys = {
        "device",
        "data_root",
        "save_dir",
        "root_dir",
        "num_workers",
        "num_splits",
        "molecule_cols",
        "scalar_cols",
        "label_col",
        "task",
        "scalar_scale",
        "config_str",
        "fine_tune_from",
        "log_every_n_steps",
        "eval_every_n_epochs",
        "train_ratio",
        'valid_ratio',
        'test_ratio',
        'dataset_name',
        'scalar_hidden_dim',
        'pred_hidden_dim',
    }
    parts = []
    for key, value in vars(args).items():
        if key in ignore_keys:
            continue
        parts.append(str(value))
    return "_".join(parts)


def _resolve_dataset_paths(dataset_name, additional_str, ratio_dir):
    """Return dataset root and result root for a supported dataset."""
    dataset_dir = LEGACY_DATASET_DIRS.get(dataset_name, dataset_name)
    form_prefix = LEGACY_FORM_PREFIXES.get(dataset_name, dataset_name)
    form_dir = f"{form_prefix}_form_screen{additional_str}"

    data_root = os.path.join(ROOT_DIR, "dataset", dataset_dir, form_dir, ratio_dir)
    save_root = os.path.join(ROOT_DIR, "results", dataset_dir, form_dir, ratio_dir)
    return data_root, save_root


def build_parser():
    """Build the argparse interface for AGILE multi-molecule finetuning."""
    parser = argparse.ArgumentParser(description="Train AGILE with argparse configs.")
    parser.add_argument("--dataset_name", type=str, default="a549", help="Dataset name.")
    parser.add_argument("--model", type=str, choices=["interaction_transformer"], default="interaction_transformer", help="Model name.")
    parser.add_argument("--device", type=int, default=6, help="CUDA device index or string.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    parser.add_argument('--unseen_ILs', action='store_true')
    parser.add_argument('--unseen_HLs', action='store_true')
    parser.add_argument('--unseen_split_ILs'  , action='store_true') 
    parser.add_argument('--unseen_split_HLs'  , action='store_true') 
    parser.add_argument('--unseen_split_CHOLs', action='store_true') 
    parser.add_argument('--unseen_split_PEGs' , action='store_true') 
    parser.add_argument('--unseen_split_CARs' , action='store_true') 
    
    parser.add_argument("--wandb_name", type=str, default="", help="wandb log name")
    
    
    parser.add_argument("--heads", type=int, default=8, help="Batch size.")
    parser.add_argument('--pos_enc', type=str, choices=['None','rand','abs','RoPE','RiPE'], default='None', help='reserved positional encoding flag; interaction_transformer now ignores positional encodings')
    parser.add_argument("--percent_embed_dim", type=int, default=128, help="GBF embedding dim for ratio token features.")
    parser.add_argument("--component_types_embed_dim", type=int, default=128, help="Component type embedding dim for ratio-aware tokens.")
    parser.add_argument("--activation_fn", type=str, default="gelu", help="Activation used in the ratio-aware token projection.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=512, help="Graph feature dim.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num_splits", type=int, default=10, help="Number of splits to run.")
    parser.add_argument("--drop", type=float, default=0.3, help="Dropout ratio.")
    parser.add_argument("--train_ratio", type=float, default=60, help="Train ratio.")
    parser.add_argument("--valid_ratio", type=float, default=20, help="Valid ratio.")
    parser.add_argument("--test_ratio", type=float, default=20, help="Test ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--scalar_scale", type=float, default=1.0, help="Scale divisor for scalar inputs.")
    parser.add_argument("--patience", type=int, default=-1, help="Early stopping patience in epochs.")
    parser.add_argument('--consistency', type=float, default=1.0, help="must be >0 to get emb1/emb2 from model")

    parser.add_argument(
        "--molecule_cols",
        nargs="+",
        default=["IL_SMILES", "HL_SMILES", "CHOL_SMILES", "PEG_SMILES"],
        help="Molecule SMILES column names.",
    )
    parser.add_argument(
        "--scalar_cols",
        nargs="+",
        default=["mRNA_weight", "IL_ratio", "HL_ratio", "CHOL_ratio", "PEG_ratio"],
        help="Scalar feature column names.",
    )
    
    
    return parser


def main():
    """Entry point for multi-molecule AGILE finetuning."""
    parser = build_parser()
    args = parser.parse_args()
    args.train_ratio /= 100
    args.valid_ratio /= 100
    args.test_ratio /= 100
    args.init_lr = args.lr
    args.freeze_ME = True
    
    args.root_dir = ROOT_DIR
    args.device = _device_string(args.device)
    args.task = "regression"
    args.num_molecules = len(args.molecule_cols)
    
    args.emb_dim = 300
    args.label_col = 'label'
    args.pool = "mean"
    args.dropout = args.drop

    ratio_dir = f"{args.train_ratio}_{args.valid_ratio}_{args.test_ratio}"
    additional_str = ''
    if args.unseen_ILs:
        additional_str += '_usIL'
    elif args.unseen_HLs:
        additional_str += '_usHL'
    elif args.unseen_split_ILs:
        additional_str += '_usILratio'
    elif args.unseen_split_HLs:
        additional_str += '_usHLratio'
    elif args.unseen_split_CHOLs:
        additional_str += '_usCHOLratio'
    elif args.unseen_split_PEGs:
        additional_str += '_usPEGratio'
    elif args.unseen_split_CARs:
        additional_str += '_usCARratio'
    args.data_root, save_root = _resolve_dataset_paths(args.dataset_name, additional_str, ratio_dir)
    args.config_str = _get_config_string(args)
    args.save_dir = os.path.join(save_root, args.config_str)
    run_training(args)


if __name__ == "__main__":
    main()
