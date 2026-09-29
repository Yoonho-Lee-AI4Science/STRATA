import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimol_dataset import MultiMolTSVDataset, multimol_collate_fn
from agile_finetune_comet_pos import interaction_transformer

try:
    import wandb
except ImportError:
    wandb = None


torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass


LEGACY_FORM_PREFIXES = {
    "a549": "A549",
    "xh25": "XH25",
    "yz22": "YZ22",
    "yz24": "YZ24",
}


def _get_form_prefix(dataset_name):
    return LEGACY_FORM_PREFIXES.get(dataset_name, dataset_name)


def _resolve_split_dir(args, split_idx):
    form_prefix = _get_form_prefix(args.dataset_name)
    return os.path.join(args.data_root, f"split_{split_idx}", f"{form_prefix}_form_screen")


def _build_loaders(train_path, valid_path, test_path, args):
    train_dataset = MultiMolTSVDataset(
        train_path,
        args.molecule_cols,
        args.scalar_cols,
        args.label_col
    )
    valid_dataset = MultiMolTSVDataset(
        valid_path,
        args.molecule_cols,
        args.scalar_cols,
        args.label_col
    )
    test_dataset = MultiMolTSVDataset(
        test_path,
        args.molecule_cols,
        args.scalar_cols,
        args.label_col
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=multimol_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=multimol_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=multimol_collate_fn,
    )
    return train_loader, valid_loader, test_loader


def _build_model(args):
    if not hasattr(args, "percent_embed_dim"):
        args.percent_embed_dim = 128
    if not hasattr(args, "component_types_embed_dim"):
        args.component_types_embed_dim = 128
    if not hasattr(args, "activation_fn"):
        args.activation_fn = "gelu"
    if args.model == "interaction_transformer":
        return interaction_transformer(args)
    raise ValueError(f"Unsupported model: {args.model}")


def _load_pretrained_weights(model, args):
    checkpoint_path = os.path.join(args.root_dir, "checkpoints", "MolCLR", "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Pretrained checkpoint not found: {checkpoint_path}")
        return
    if not hasattr(model, "load_backbone_state_dict"):
        return
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    model.load_backbone_state_dict(state_dict)


def _freeze_molecule_encoders(model):
    for attr in ["molecule_encoder", "Condtional_ME"]:
        module = getattr(model, attr, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False




def _build_adamw_param_groups(model, weight_decay):
    conditional_prefixes = (
        "Condtional_ME.",
        "molecule_encoder.",
        "learnable_molecule_encoder.",
    )
    conditional_decay_params = []
    other_decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        elif name.startswith(conditional_prefixes):
            conditional_decay_params.append(param)
        else:
            other_decay_params.append(param)

    param_groups = []
    if conditional_decay_params:
        param_groups.append({"params": conditional_decay_params, "weight_decay": 0.01})
    if other_decay_params:
        param_groups.append({"params": other_decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def _build_optimizer_and_scheduler(model, args):
    param_groups = _build_adamw_param_groups(model, args.wd)
    if not param_groups:
        raise RuntimeError("No trainable parameters found when building optimizer.")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    return optimizer, None


def _autocast_context(device):
    return nullcontext()


def _unpack_model_outputs(outputs):
    if not isinstance(outputs, tuple):
        return outputs, None, None, None
    if len(outputs) == 4:
        return outputs
    if len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2], None
    if len(outputs) == 2:
        return outputs[0], outputs[1], None, None
    if len(outputs) == 1:
        return outputs[0], None, None, None
    raise ValueError(f"Unexpected number of model outputs: {len(outputs)}")


def _unpack_embedding_and_mask(emb):
    if emb is None:
        return None, None
    if isinstance(emb, tuple):
        tensor, mask = emb
    else:
        tensor, mask = emb, None
    if mask is not None:
        mask = mask.to(device=tensor.device, dtype=tensor.dtype)
    return tensor, mask


def _masked_mean(values, mask):
    if mask is None:
        return values.mean()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _consistency_loss(args, emb1, emb2):
    if args.consistency <= 0 or emb1 is None or emb2 is None:
        return None
    emb1, mask1 = _unpack_embedding_and_mask(emb1)
    emb2, mask2 = _unpack_embedding_and_mask(emb2)
    if emb1 is None or emb2 is None:
        return None
    valid_mask = None
    if mask1 is not None and mask2 is not None:
        valid_mask = mask1 * mask2
    elif mask1 is not None:
        valid_mask = mask1
    elif mask2 is not None:
        valid_mask = mask2
    if valid_mask is not None and valid_mask.sum() <= 0:
        return None

    additional_loss = 0.0
    has_term = False

    mse12 = ((emb1.float().detach() - emb2.float()) ** 2).mean(dim=-1)
    mse21 = ((emb1.float() - emb2.float().detach()) ** 2).mean(dim=-1)
    consistency_loss_mse = (_masked_mean(mse12, valid_mask) + _masked_mean(mse21, valid_mask))
    additional_loss = additional_loss + args.consistency * consistency_loss_mse/ 2
    has_term = True

    return additional_loss if has_term else None


def _train_epoch(model, loader, optimizer, device, args):
    model.train()
    train_pred_loss = 0.0
    train_additional_loss = 0.0
    train_count = 0

    for mol_batch, scalar_batch, labels, _ in loader:
        mol_batch = mol_batch.to(device)
        scalar_batch = scalar_batch.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            output, output2, emb1, emb2 = _unpack_model_outputs(model(mol_batch, scalar_batch))
            pred_loss = F.mse_loss(output.float(), labels)
            #if output2 is not None:
            #    pred_loss = pred_loss + F.mse_loss(output2.float(), labels)
            additional_loss = _consistency_loss(args, emb1, emb2)
            total_loss = pred_loss if additional_loss is None else pred_loss + additional_loss

        total_loss.backward()
        optimizer.step()

        train_pred_loss += pred_loss.item() * labels.numel()
        if additional_loss is not None:
            train_additional_loss += additional_loss.item() * labels.numel()
        train_count += labels.numel()

    train_pred_loss_mean = train_pred_loss / max(train_count, 1)
    train_additional_loss_mean = train_additional_loss / max(train_count, 1)
    train_loss_mean = train_pred_loss_mean + train_additional_loss_mean
    return train_loss_mean, train_pred_loss_mean, train_additional_loss_mean


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    sse = 0.0
    sample_count = 0
    preds_list = []
    labels_list = []
    lnp_ids = []

    for mol_batch, scalar_batch, labels, batch_ids in loader:
        mol_batch = mol_batch.to(device)
        scalar_batch = scalar_batch.to(device)
        labels = labels.float().to(device)

        with _autocast_context(device):
            output, _, _, _ = _unpack_model_outputs(model(mol_batch, scalar_batch))
        preds = output.float()

        sse += torch.sum((preds - labels) ** 2).item()
        sample_count += labels.numel()
        preds_list.append(preds.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        lnp_ids.extend(batch_ids)

    mse = sse / max(sample_count, 1)
    preds = np.concatenate(preds_list, axis=0).reshape(-1)
    labels = np.concatenate(labels_list, axis=0).reshape(-1)
    return mse, preds, labels, lnp_ids


def _save_test_results(save_path, lnp_ids, labels, preds):
    result_path = os.path.join(save_path, "result.txt")
    if len(lnp_ids) != len(labels) or len(labels) != len(preds):
        raise ValueError("Result lengths do not match for lnp_ids/labels/preds.")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in lnp_ids) + "\n")
        f.write(" ".join(str(v) for v in labels) + "\n")
        f.write(" ".join(str(v) for v in preds) + "\n")


def train_single_split(split_idx, args):
    use_wandb = args.wandb_name != "" and split_idx == 0
    split_dir = _resolve_split_dir(args, split_idx)
    train_path = os.path.join(split_dir, "train.tsv")
    valid_path = os.path.join(split_dir, "valid.tsv")
    test_path = os.path.join(split_dir, "test.tsv")

    save_path = os.path.join(args.save_dir, f"split_{split_idx}")
    os.makedirs(save_path, exist_ok=True)

    train_loader, valid_loader, test_loader = _build_loaders(train_path, valid_path, test_path, args)

    model = _build_model(args).to(args.device)
    _load_pretrained_weights(model, args)
    if args.freeze_ME:
        _freeze_molecule_encoders(model)

    optimizer, scheduler = _build_optimizer_and_scheduler(model, args)

    best_valid_mse = float("inf")
    best_epoch = -1
    best_model_path = os.path.join(save_path, "model.pth")
    patience_counter = 0

    epoch_pbar = tqdm(range(args.epochs), desc=f"Split {split_idx}")
    for epoch in epoch_pbar:
        train_loss, train_pred_loss, train_additional_loss = _train_epoch(
            model, train_loader, optimizer, args.device, args
        )
        valid_mse, _, _, _ = _evaluate(model, valid_loader, args.device)
        test_mse, _, _, _ = _evaluate(model, test_loader, args.device)

        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.6f}",
            valid_mse=f"{valid_mse:.6f}",
            test_mse=f"{test_mse:.6f}",
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            log_payload = {
                "train_loss": train_loss,
                "train_pred_loss": train_pred_loss,
                "train_additional_loss": train_additional_loss,
                "valid_mse": valid_mse,
                "test_mse": test_mse,
                "lr": current_lr,
            }
            if train_loss > 0:
                log_payload["train_pred_percent"] = train_pred_loss * 100.0 / train_loss
                log_payload["train_additional_percent"] = train_additional_loss * 100.0 / train_loss
            wandb.log(log_payload, step=epoch)

        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_model_path)
            patience_counter = 0
        elif args.patience > 0 and epoch > args.patience:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(checkpoint["model"])

    test_mse, preds, labels, lnp_ids = _evaluate(model, test_loader, args.device)
    test_rmse = float(np.sqrt(test_mse))
    print(f"Split {split_idx} | Best epoch: {best_epoch} | Test MSE: {test_mse:.6f} | RMSE: {test_rmse:.6f}")
    if use_wandb and args.epochs > 0:
        wandb.log({"test_mse_final": test_mse}, step=args.epochs)
    _save_test_results(save_path, lnp_ids, labels, preds)
    return test_mse, test_rmse


def run_training(args):
    use_wandb = args.wandb_name != ""
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when args.wandb_name is not empty.")
        wandb.init(project=args.wandb_name, name=args.config_str)

    test_mse_list = []
    test_rmse_list = []
    try:
        for split_idx in tqdm(range(args.num_splits), desc="Splits"):
            test_mse, test_rmse = train_single_split(split_idx, args)
            test_mse_list.append(test_mse)
            test_rmse_list.append(test_rmse)
    finally:
        if use_wandb:
            wandb.finish()

    if test_mse_list:
        print(
            f"{args.config_str} || test_mse mean: {np.mean(test_mse_list):.6f}, std: {np.std(test_mse_list):.6f}"
        )
