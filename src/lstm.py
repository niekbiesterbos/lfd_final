import argparse
import json
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from data_utils import load_splits, to_xy
from metrics import full_report
from common import ensure_dir, set_seed
from tqdm import tqdm
import numpy as np


class TextDataset(Dataset):
    """Padded index dataset built from a fitted vocabulary and tokenization function."""

    def __init__(self, texts: List[str], labels: List[int], vocab, tokenizer, max_len: int):
        self.labels = labels
        self.max_len = max_len
        self.vocab = vocab
        self.tok = tokenizer
        self.text_ids = [self.encode(t) for t in texts]

    def encode(self, text: str):
        tokens = self.tok(text)
        ids = [self.vocab.get(tok, self.vocab["<unk>"])
               for tok in tokens][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.text_ids[idx], torch.tensor(
        self.labels[idx], dtype=torch.long)


class BiLSTM(nn.Module):
    """BiLSTM classifier over static embeddings with mean pooling."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int, dropout: float, num_classes: int, embeddings=None, freeze_emb: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(torch.tensor(embeddings))
        self.emb.weight.requires_grad = not freeze_emb
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        # Mean pooling over time
        pooled = out.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


def simple_tokenizer(text: str) -> List[str]:
    """Lightweight tokenizer for tweets: lowercasing + whitespace split."""
    return text.lower().strip().split()


def build_vocab(texts: List[str], min_freq: int = 2) -> dict:
    """Create token -> index mapping with PAD and UNK; rare tokens mapped to UNK."""
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(simple_tokenizer(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, f in cnt.items():
        if f >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def load_static_embeddings(vocab: dict, path_txt: str, dim: int) -> Tuple[List[List[float]], int]:
    """Load GloVe/FastText text embeddings into a matrix aligned with vocab; OOV tokens get random vectors."""

    emb = np.random.normal(scale=0.02, size=(
        len(vocab), dim)).astype("float32")
    with open(path_txt, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            token, vec = parts[0], parts[1:]
            if token in vocab and len(vec) == dim:
                emb[vocab[token]] = np.array(
                    list(map(float, vec)), dtype="float32")
    return emb, dim


def train_epoch(model, loader, criterion, optimizer, device, epoch: int) -> float:
    """Run one epoch of training with progress bar and return mean loss."""
    model.train()
    total_loss = 0.0

    # Display progress bar for each batch
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        progress.set_postfix(loss=loss.item())

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gold = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        gold.extend(y.tolist())
    return preds, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--outdir", default="artifacts/lstm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb-path", required=True,
                    help="Path to txt embeddings (e.g., glove.6B.100d.txt)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--freeze-emb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    tr, dv, te = load_splits(args.train, args.dev,
                             args.test)
    X_tr, y_tr = to_xy(tr)
    X_dv, y_dv = to_xy(dv)
    X_te, y_te = to_xy(te)

    vocab = build_vocab(X_tr, min_freq=args.min_freq)
    emb_matrix, _ = load_static_embeddings(vocab, args.emb_path, 100)

    ds_tr = TextDataset(X_tr, y_tr, vocab, simple_tokenizer, args.max_len)
    ds_dv = TextDataset(X_dv, y_dv, vocab, simple_tokenizer, args.max_len)
    ds_te = TextDataset(X_te, y_te, vocab, simple_tokenizer, args.max_len)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_dv = DataLoader(ds_dv, batch_size=args.batch_size)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM(vocab_size=len(vocab), emb_dim=100, hidden=args.hidden,
                   num_layers=args.layers, dropout=args.dropout, num_classes=2,
                   embeddings=emb_matrix, freeze_emb=args.freeze_emb).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, dl_tr, criterion, optimizer, device, epoch)
        pred_dv, gold_dv = predict(model, dl_dv, device)
        rep = full_report(gold_dv, pred_dv)
        f1 = rep["macro_f1"]
        Path(f"{args.outdir}/dev_epoch{epoch}.json").write_text(json.dumps(rep, indent=2))
        if f1 > best_f1:
            best_f1 = f1
            best_path = f"{args.outdir}/best.pt"
            torch.save({"model": model.state_dict(),
                       "vocab": vocab}, best_path)

    # Test with best checkpoint
    chk = torch.load(best_path, map_location="cpu")
    model.load_state_dict(chk["model"])
    pred_te, gold_te = predict(model, dl_te, device)
    test_rep = full_report(gold_te, pred_te)
    Path(f"{args.outdir}/test_metrics.json").write_text(json.dumps(test_rep, indent=2))


if __name__ == "__main__":
    main()
