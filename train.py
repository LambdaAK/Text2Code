"""Training script for NL → code Transformer."""

import argparse
import json
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


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cross-entropy loss, ignoring padding positions."""
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask = targets_flat != pad_id
    loss = nn.functional.cross_entropy(
        logits_flat[mask], targets_flat[mask], reduction="mean"
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
) -> tuple[float, float]:
    """Compute loss and exact match accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for nl, code_in, code_out, _, _ in loader:
        nl = nl.to(device)
        code_in = code_in.to(device)
        code_out = code_out.to(device)

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
) -> float:
    """One training epoch."""
    model.train()
    total_loss = 0.0
    n = 0

    for nl, code_in, code_out, _, _ in loader:
        nl = nl.to(device)
        code_in = code_in.to(device)
        code_out = code_out.to(device)

        optimizer.zero_grad()
        logits = model(nl, code_in)
        loss = compute_loss(logits, code_out, pad_id)
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint.pt")
    parser.add_argument("--max-nl-len", type=int, default=64)
    parser.add_argument("--max-code-len", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSeq2Seq(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, pad_id, device)
        val_loss, val_acc = evaluate(model, val_loader, vocab, pad_id, eos_id, device)

        print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "vocab_size": len(vocab),
                    "vocab_itos": [vocab.itos[i] for i in range(len(vocab))],
                    "config": vars(args),
                },
                args.save,
            )
            print(f"  → saved checkpoint to {args.save}")

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
