import json
import os
import random
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


PANTRY_BUCKET_SPLIT = [0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]


def require_file(path, description="file"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_dir(path, description="directory"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def pantry_paths(args):
    data_dir = Path(args.data_dir)
    return {
        "data_dir": data_dir,
        "metadata": data_dir / "metadata_Prime_Pantry.csv",
        "history": data_dir / "Prime_Pantry_cleaned_history.csv",
        "frequency": data_dir / "Prime_Pantry_fre.csv",
        "llm2vec": Path(args.llm2vec_output),
    }


def load_pantry_metadata(args):
    metadata_path = require_file(pantry_paths(args)["metadata"], "Prime Pantry metadata")
    metadata = pd.read_csv(metadata_path)
    if "asin" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain an 'asin' column")
    metadata = metadata.copy()
    metadata["index"] = metadata.index + 1
    item_name_to_id = metadata.set_index("asin")["index"].to_dict()
    return metadata, item_name_to_id


def get_pantry_item_num(args):
    metadata, _ = load_pantry_metadata(args)
    return int(len(metadata))


def validate_item_embeddings(item_word_embs, item_num, path):
    expected = item_num + 1
    actual = int(item_word_embs.shape[0])
    if actual != expected:
        raise ValueError(
            f"LLM2Vec embedding row count mismatch for {path}: expected {expected} "
            f"(item_num + 1), got {actual}"
        )


def build_pantry_frequency_bins(args, item_num, log_fn=None):
    fre_path = require_file(pantry_paths(args)["frequency"], "Prime Pantry frequency file")
    fre = pd.read_csv(fre_path)
    required = {"id", "fre"}
    missing = required.difference(fre.columns)
    if missing:
        raise ValueError(f"{fre_path} must contain columns {sorted(required)}; missing {sorted(missing)}")

    ids = fre["id"].astype(int)
    expected_ids = set(range(item_num + 1))
    actual_ids = set(ids.tolist())
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)[:10]
        extra_ids = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            f"{fre_path} id range must align with 0..{item_num}; "
            f"missing sample={missing_ids}, extra sample={extra_ids}"
        )
    if PANTRY_BUCKET_SPLIT[-1] != item_num + 1:
        raise ValueError(
            f"Prime Pantry bucket split ends at {PANTRY_BUCKET_SPLIT[-1]}, "
            f"but item_num + 1 is {item_num + 1}"
        )

    ranked = fre.sort_values("fre", ascending=False).copy()
    ranked["bin"] = 0
    for bucket in range(10):
        ranked.iloc[PANTRY_BUCKET_SPLIT[bucket]:PANTRY_BUCKET_SPLIT[bucket + 1], ranked.columns.get_loc("bin")] = bucket + 1
    ranked = ranked.sort_values("id", ascending=True)
    bins = ranked["bin"].to_numpy()
    bins[0] = 0
    counts = {bucket: int((bins == bucket).sum()) for bucket in range(11)}
    message = "Prime Pantry bucket counts: " + ", ".join(f"{k}:{v}" for k, v in counts.items())
    if log_fn is not None:
        log_fn(message)
    else:
        print(message)
    return bins


def set_seed(seed, rank=0):
    full_seed = int(seed) + int(rank)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)
    np.random.seed(full_seed)
    random.seed(full_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id, seed, rank):
    worker_seed = int(seed) + int(rank) * 1000 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def configure_batch_size(args, world_size, log_fn=print):
    if args.auto_adjust_batch_size:
        if args.target_global_batch_size % world_size != 0:
            warnings.warn(
                f"target_global_batch_size={args.target_global_batch_size} is not divisible by "
                f"world_size={world_size}; effective batch will be smaller than target.",
                stacklevel=2,
            )
        args.per_device_batch_size = max(1, args.target_global_batch_size // world_size)
        args.batch_size = args.per_device_batch_size
    local_global_batch = int(args.per_device_batch_size) * int(world_size)
    if args.gradient_accumulation_steps is None:
        if args.target_global_batch_size % local_global_batch != 0:
            warnings.warn(
                f"target_global_batch_size={args.target_global_batch_size} is not divisible by "
                f"per_device_batch_size*world_size={local_global_batch}; effective batch will be "
                f"{local_global_batch}.",
                stacklevel=2,
            )
            args.gradient_accumulation_steps = 1
        else:
            args.gradient_accumulation_steps = max(1, args.target_global_batch_size // local_global_batch)
    args.effective_global_batch_size = local_global_batch * int(args.gradient_accumulation_steps)
    log_fn(f"world_size={world_size}")
    log_fn(f"per_device_batch_size={args.per_device_batch_size}")
    log_fn(f"gradient_accumulation_steps={args.gradient_accumulation_steps}")
    log_fn(f"effective_global_batch_size={args.effective_global_batch_size}")
    log_fn(f"eval_batch_size={args.eval_batch_size}")
    return args


def phase_checkpoint_dir(args, phase):
    return Path(args.checkpoint_dir) / phase


def phase_output_dir(args, phase):
    return Path(args.output_dir) / phase


def prepare_run_dirs(args, phase):
    ckpt_dir = phase_checkpoint_dir(args, phase)
    out_dir = phase_output_dir(args, phase)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.run_checkpoint_dir = ckpt_dir
    args.run_output_dir = out_dir
    return ckpt_dir, out_dir


def list_checkpoints(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.glob("*.pt"))


def resolve_load_checkpoint(args, phase):
    ckpt_name = args.load_ckpt_name
    if ckpt_name in (None, "", "None"):
        return None

    if phase == "phase2" and args.phase1_ckpt_dir is not None:
        directory = Path(args.phase1_ckpt_dir)
    elif phase == "phase2":
        directory = phase_checkpoint_dir(args, "phase1")
    elif phase == "phase3" and args.phase2_ckpt_dir is not None:
        directory = Path(args.phase2_ckpt_dir)
    elif phase == "phase3":
        directory = phase_checkpoint_dir(args, "phase2")
    else:
        directory = phase_checkpoint_dir(args, phase)

    ckpt_path = directory / ckpt_name
    print(f"Loading checkpoint path: {ckpt_path}")
    if not ckpt_path.exists():
        available = list_checkpoints(directory)
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Available checkpoints in {directory}: {available}"
        )
    return ckpt_path


def save_checkpoint(now_epoch, model, model_dir, optimizer, scaler, log_fn, best_name=None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "scaler_state": scaler.state_dict(),
        "epoch": now_epoch,
    }
    epoch_path = model_dir / f"epoch-{now_epoch}.pt"
    torch.save(state, epoch_path)
    log_fn(f"Model saved to {epoch_path}")
    if best_name is not None:
        best_path = model_dir / best_name
        shutil.copy2(epoch_path, best_path)
        log_fn(f"Best model saved to {best_path}")
        return best_path
    return epoch_path


def load_checkpoint_into_model(model, ckpt_path, local_rank, strict=False):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    return checkpoint


def load_best_checkpoint_for_eval(args, phase, model, local_rank, log_fn=print):
    best_path = phase_checkpoint_dir(args, phase) / args.save_best_name
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found for {phase}: {best_path}")
    target = model.module if hasattr(model, "module") else model
    checkpoint = load_checkpoint_into_model(target, best_path, local_rank, strict=False)
    log_fn(f"Loaded best checkpoint for final test: {best_path}")
    return checkpoint, best_path


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_args_json(args):
    path = Path(args.run_output_dir) / "args.json"
    data = {key: jsonable(value) for key, value in vars(args).items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def append_metrics(args, record):
    path = Path(args.run_output_dir) / "metrics.jsonl"
    clean = {key: jsonable(value) for key, value in record.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean, sort_keys=True) + "\n")


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def save_best_summary(args, summary):
    path = Path(args.run_output_dir) / "best_summary.json"
    clean = {key: jsonable(value) for key, value in summary.items()}
    clean.setdefault("git_commit_hash", git_commit_hash())
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, sort_keys=True)
    return path
