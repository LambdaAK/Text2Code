"""Dataset and tokenization for (NL, code) pairs."""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# Special tokens
PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

# Code tokenization: match operators (longest first), then numbers, identifiers, punctuation
_CODE_PATTERN = re.compile(
    r"==|!=|&&|\|\||\d+|[a-zA-Z]+|[()+=\-*<>&|.]"
)


def tokenize_nl(text: str) -> List[str]:
    """Tokenize natural language by splitting on whitespace."""
    return text.lower().split()


def tokenize_code(code: str) -> List[str]:
    """Tokenize code into operators, identifiers, numbers, and punctuation."""
    tokens = _CODE_PATTERN.findall(code)
    return [t for t in tokens if t.strip()]


class Vocab:
    """Vocabulary for encoding/decoding token sequences."""

    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.special = special_tokens or [PAD, UNK, SOS, EOS]
        self.stoi = {t: i for i, t in enumerate(self.special)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self._frozen = False

    def add(self, token: str) -> int:
        if token not in self.stoi:
            if self._frozen:
                return self.stoi[UNK]
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token
            return idx
        return self.stoi[token]

    def freeze(self):
        self._frozen = True

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi[UNK]) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos.get(i, UNK) for i in ids]


def build_vocab(pairs: List[Tuple[str, str]]) -> Vocab:
    """Build vocabulary from (nl, code) pairs."""
    vocab = Vocab()
    for nl, code in pairs:
        for t in tokenize_nl(nl):
            vocab.add(t)
        for t in tokenize_code(code):
            vocab.add(t)
    vocab.freeze()
    return vocab


class NLCodeDataset(Dataset):
    """Dataset of (natural language, code) pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        vocab: Vocab,
        max_nl_len: int = 64,
        max_code_len: int = 64,
    ):
        self.pairs = pairs
        self.vocab = vocab
        self.max_nl_len = max_nl_len
        self.max_code_len = max_code_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nl, code = self.pairs[idx]
        nl_tokens = tokenize_nl(nl)[: self.max_nl_len]
        code_tokens = tokenize_code(code)[: self.max_code_len]

        nl_ids = [self.vocab.stoi[SOS]] + self.vocab.encode(nl_tokens) + [self.vocab.stoi[EOS]]
        code_ids = [self.vocab.stoi[SOS]] + self.vocab.encode(code_tokens) + [self.vocab.stoi[EOS]]

        nl_tensor = torch.tensor(nl_ids, dtype=torch.long)
        code_in = torch.tensor(code_ids[:-1], dtype=torch.long)
        code_out = torch.tensor(code_ids[1:], dtype=torch.long)

        return nl_tensor, code_in, code_out


def load_pairs(path: str) -> List[Tuple[str, str]]:
    """Load (nl, code) pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["nl"], obj["code"]))
    return pairs


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """Pad sequences to same length within batch."""
    nl_list, code_in_list, code_out_list = zip(*batch)
    nl_lens = [x.size(0) for x in nl_list]
    code_lens = [x.size(0) for x in code_in_list]

    nl_padded = torch.nn.utils.rnn.pad_sequence(nl_list, batch_first=True, padding_value=pad_id)
    code_in_padded = torch.nn.utils.rnn.pad_sequence(code_in_list, batch_first=True, padding_value=pad_id)
    code_out_padded = torch.nn.utils.rnn.pad_sequence(code_out_list, batch_first=True, padding_value=pad_id)

    return nl_padded, code_in_padded, code_out_padded, nl_lens, code_lens
