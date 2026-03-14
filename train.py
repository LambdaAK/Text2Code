"""Training script for NL → code Transformer.

Optimized for A100 GPU: mixed precision (BF16), parallel data loading,
larger batch sizes, and optional torch.compile.
"""

import argparse
import itertools
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    build_vocab,
    collate_fn,
    load_pairs,
    NLCodeDataset,
)
from model import TransformerSeq2Seq


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss, ignoring padding positions."""
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask = targets_flat != pad_id
    loss = nn.functional.cross_entropy(
        logits_flat[mask],
        targets_flat[mask],
        reduction="mean",
        label_smoothing=label_smoothing,
    )
    return loss


def exact_match(pred_ids: list, gold_ids: list) -> bool:
    """Check if predicted sequence matches gold (excluding SOS/EOS)."""
    # Strip SOS/EOS (first and last)
    pred = pred_ids[1:-1] if len(pred_ids) > 2 else pred_ids
    gold = gold_ids[1:-1] if len(gold_ids) > 2 else gold_ids
    return pred == gold


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Compute loss and exact match accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    non_blocking = loader.pin_memory if hasattr(loader, "pin_memory") else False

    for nl, code_in, code_out, _, _ in loader:
        nl = nl.to(device, non_blocking=non_blocking)
        code_in = code_in.to(device, non_blocking=non_blocking)
        code_out = code_out.to(device, non_blocking=non_blocking)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(nl, code_in)
                loss = compute_loss(logits.float(), code_out, pad_id)
        else:
            logits = model(nl, code_in)
            loss = compute_loss(logits, code_out, pad_id)
        total_loss += loss.item() * nl.size(0)
        total_tokens += (code_out != pad_id).sum().item()

        # Greedy decode for exact match
        pred_ids = logits.argmax(dim=-1)
        for i in range(nl.size(0)):
            pred_seq = pred_ids[i].tolist()
            gold_seq = code_out[i].tolist()
            # Truncate at EOS
            try:
                eos_idx = gold_seq.index(eos_id)
                gold_seq = gold_seq[: eos_idx + 1]
            except ValueError:
                pass
            try:
                eos_idx = pred_seq.index(eos_id)
                pred_seq = pred_seq[: eos_idx + 1]
            except ValueError:
                pass
            if exact_match(pred_seq, gold_seq):
                total_correct += 1

    n = len(loader.dataset)
    avg_loss = total_loss / n
    acc = total_correct / n
    return avg_loss, acc


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    device: torch.device,
    use_amp: bool = False,
    label_smoothing: float = 0.0,
) -> float:
    """One training epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    non_blocking = loader.pin_memory if hasattr(loader, "pin_memory") else False

    for nl, code_in, code_out, _, _ in loader:
        nl = nl.to(device, non_blocking=non_blocking)
        code_in = code_in.to(device, non_blocking=non_blocking)
        code_out = code_out.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(nl, code_in)
                loss = compute_loss(logits.float(), code_out, pad_id, label_smoothing=label_smoothing)
        else:
            logits = model(nl, code_in)
            loss = compute_loss(logits, code_out, pad_id, label_smoothing=label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * nl.size(0)
        n += nl.size(0)

    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train.jsonl", help="Training data (JSONL)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction for validation")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs (0 = train indefinitely until Ctrl+C)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (A100: 128-512)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size preset (small=256d/3L, medium=384d/4L, large=512d/6L)")
    parser.add_argument("--d-model", type=int, default=None, help="Override model dimension")
    parser.add_argument("--nhead", type=int, default=None, help="Override attention heads")
    parser.add_argument("--num-layers", type=int, default=None, help="Override encoder/decoder layers")
    parser.add_argument("--dim-ff", type=int, default=None, help="Override FFN dimension")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (default: 0.1/0.12/0.15 for small/medium/large)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing (reduces overfitting)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"],
                        help="LR schedule: cosine=warm restarts every 20 epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint.pt")
    parser.add_argument("--max-nl-len", type=int, default=64)
    parser.add_argument("--max-code-len", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (parallel loading)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (BF16)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    args = parser.parse_args()

    # Model size presets (overridden by explicit --d-model etc.)
    PRESETS = {
        "small":  {"d_model": 256, "nhead": 4,  "num_layers": 3, "dim_ff": 512, "dropout": 0.1},
        "medium": {"d_model": 384, "nhead": 6,  "num_layers": 4, "dim_ff": 1024, "dropout": 0.12},
        "large":  {"d_model": 512, "nhead": 8,  "num_layers": 6, "dim_ff": 2048, "dropout": 0.15},
    }
    preset = PRESETS[args.model_size]
    d_model = args.d_model if args.d_model is not None else preset["d_model"]
    nhead = args.nhead if args.nhead is not None else preset["nhead"]
    num_layers = args.num_layers if args.num_layers is not None else preset["num_layers"]
    dim_ff = args.dim_ff if args.dim_ff is not None else preset["dim_ff"]
    dropout = args.dropout if args.dropout is not None else preset["dropout"]

    torch.manual_seed(args.seed)
    print(f"Model: {args.model_size} (d_model={d_model}, nhead={nhead}, layers={num_layers}, dim_ff={dim_ff}, dropout={dropout})")

    # Load data
    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} pairs from {args.data}")

    # Split train/val
    indices = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]

    # Build vocab and datasets
    vocab = build_vocab(pairs)
    pad_id = vocab.stoi["<pad>"]
    eos_id = vocab.stoi["<eos>"]
    print(f"Vocabulary size: {len(vocab)}")

    train_ds = NLCodeDataset(
        train_pairs, vocab, max_nl_len=args.max_nl_len, max_code_len=args.max_code_len
    )
    val_ds = NLCodeDataset(
        val_pairs, vocab, max_nl_len=args.max_nl_len, max_code_len=args.max_code_len
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"
    num_workers = args.num_workers if device.type == "cuda" else 0  # workers + pin_memory only useful with GPU

    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    if use_amp:
        print("Using mixed precision (BF16)")
    if pin_memory:
        print(f"DataLoader: {num_workers} workers, pin_memory=True")

    # Model
    model = TransformerSeq2Seq(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        pad_id=pad_id,
    ).to(device)

    try:
        from torch._inductor.exc import InductorError
    except ImportError:
        InductorError = type(None)  # never matches

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (mode=default)...")
        compiled = torch.compile(model, mode="default")
        # Compilation happens lazily on first forward; warm up to catch errors early
        try:
            batch = next(iter(train_loader))
            nl, code_in = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _ = compiled(nl, code_in)
            model = compiled
        except Exception as e:
            if "libcuda" in str(e).lower() or (InductorError and isinstance(e, InductorError)):
                print(f"torch.compile failed: {e}")
                print("Continuing without compile.")
            else:
                raise

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        print("LR schedule: cosine warm restarts (T_0=20)")

    # Training loop (epochs=0 means train indefinitely)
    best_val_acc = 0.0
    epoch_iter = range(1, args.epochs + 1) if args.epochs > 0 else itertools.count(1)
    try:
        for epoch in epoch_iter:
            try:
                train_loss = train_epoch(
                    model, train_loader, optimizer, pad_id, device,
                    use_amp=use_amp, label_smoothing=args.label_smoothing,
                )
            except Exception as e:
                if hasattr(model, "_orig_mod") and (
                    (InductorError and isinstance(e, InductorError))
                    or "inductor" in type(e).__name__.lower()
                ):
                    print(f"torch.compile failed during training: {e}")
                    print("Falling back to uncompiled model for remaining epochs.")
                    model = model._orig_mod
                    train_loss = train_epoch(
                        model, train_loader, optimizer, pad_id, device,
                        use_amp=use_amp, label_smoothing=args.label_smoothing,
                    )
                else:
                    raise
            val_loss, val_acc = evaluate(model, val_loader, vocab, pad_id, eos_id, device, use_amp=use_amp)

            lr_str = f" | lr: {optimizer.param_groups[0]['lr']:.2e}" if scheduler else ""
            print(f"Epoch {epoch:5d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.2%} | best: {best_val_acc:.2%}{lr_str}")

            if scheduler is not None:
                scheduler.step()

            is_new_best = val_acc > best_val_acc
            if is_new_best:
                best_val_acc = val_acc

            # Save every 5 epochs
            if epoch % 5 == 0:
                state_to_save = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                config = dict(vars(args))
                config.update(d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, dropout=dropout)
                ckpt = {
                    "epoch": epoch,
                    "model_state": state_to_save,
                    "optimizer_state": optimizer.state_dict(),
                    "vocab_size": len(vocab),
                    "vocab_itos": [vocab.itos[i] for i in range(len(vocab))],
                    "config": config,
                }
                torch.save(ckpt, args.save)
                best_path = args.save.replace(".pt", "_best.pt") if args.save.endswith(".pt") else args.save + "_best"
                if is_new_best:
                    torch.save(ckpt, best_path)
                    print(f"  → saved {args.save} (best → {best_path})")
                else:
                    print(f"  → saved {args.save}")
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
