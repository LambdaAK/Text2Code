"""Generate code from natural language using trained model."""

import argparse
import json

import torch

from dataset import build_vocab, load_pairs, tokenize_nl, tokenize_code
from model import TransformerSeq2Seq


def generate(
    model: TransformerSeq2Seq,
    vocab,
    nl: str,
    device: torch.device,
    max_len: int = 64,
) -> str:
    """Generate code from natural language."""
    pad_id = vocab.stoi["<pad>"]
    eos_id = vocab.stoi["<eos>"]
    sos_id = vocab.stoi["<sos>"]

    nl_tokens = tokenize_nl(nl)
    nl_ids = [sos_id] + vocab.encode(nl_tokens) + [eos_id]
    src = torch.tensor([nl_ids], dtype=torch.long, device=device)

    # Decode autoregressively
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, model._make_pad_mask(src))

        tgt_ids = [sos_id]
        for _ in range(max_len - 1):
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            logits = model.decode(
                tgt,
                memory,
                tgt_mask=model._make_causal_mask(len(tgt_ids), device),
                memory_key_padding_mask=model._make_pad_mask(src),
            )
            next_id = logits[0, -1, :].argmax().item()
            tgt_ids.append(next_id)
            if next_id == eos_id:
                break

    code_tokens = vocab.decode(tgt_ids[1:-1])  # strip SOS/EOS
    return " ".join(code_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--data", type=str, default="train.jsonl", help="Fallback: data to rebuild vocab if not in ckpt")
    parser.add_argument("--input", type=str, help="Single NL string to translate")
    parser.add_argument("--file", type=str, help="JSONL file with 'nl' field")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Rebuild vocab from checkpoint or data
    if "vocab_itos" in ckpt:
        from dataset import Vocab
        vocab = Vocab()
        for i, t in enumerate(ckpt["vocab_itos"]):
            vocab.stoi[t] = i
            vocab.itos[i] = t
        vocab._frozen = True
    else:
        pairs = load_pairs(args.data)
        vocab = build_vocab(pairs)

    config = ckpt.get("config", {})
    pad_id = vocab.stoi["<pad>"]

    model = TransformerSeq2Seq(
        vocab_size=len(vocab),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 4),
        num_encoder_layers=config.get("num_layers", 3),
        num_decoder_layers=config.get("num_layers", 3),
        dim_feedforward=config.get("dim_ff", 512),
        dropout=0.0,
        pad_id=pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if args.input:
        code = generate(model, vocab, args.input, device)
        print(code)
    elif args.file:
        with open(args.file) as f:
            for line in f:
                obj = json.loads(line)
                nl = obj["nl"]
                code = generate(model, vocab, nl, device)
                print(json.dumps({"nl": nl, "code": code}))
    else:
        print("Enter natural language (Ctrl-D to exit):")
        try:
            while True:
                nl = input("> ")
                if not nl:
                    continue
                code = generate(model, vocab, nl, device)
                print(code)
        except EOFError:
            pass


if __name__ == "__main__":
    main()
