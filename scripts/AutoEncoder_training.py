#!/usr/bin/env python3
"""Train AutoEncoder on price history tokens for a set of tickers.

This script creates a dataset per ticker, concatenates them, and trains the
AutoEncoder defined in `src/models/AutoEncoder.py`. It records per-epoch
CrossEntropy loss and an MSE between softmax probabilities and one-hot
targets (to produce an MSE-style curve), plots MSE over epochs, and saves the
trained model under `./models/autoencoder.pt` and the plot to
`./models/mse_training.png`.
"""
from pathlib import Path
import sys
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset


def ensure_paths():
    # Ensure project root and model folders are importable so existing model files
    # that perform simple `import Encoder` work as-is.
    project_root = Path(__file__).resolve().parents[1]
    src_models = project_root / "src" / "models"
    src_root = project_root / "src"
    # Insert model dir so AutoEncoder's imports like `from Encoder import Encoder`
    # find Encoder.py as a top-level module.
    sys.path.insert(0, str(src_models))
    sys.path.insert(0, str(src_root))
    sys.path.insert(0, str(project_root))


def build_dataloader(tickers: List[str], seq_len=32, batch_size=32, num_bins=256):
    from src.data.db.PriceHistoryDataset import PriceHistoryDataset

    datasets = []
    for t in tickers:
        try:
            ds = PriceHistoryDataset(symbol=t, seq_len=seq_len, transform="returns", num_bins=num_bins)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: skipping {t} due to error: {e}")

    if not datasets:
        raise RuntimeError("No datasets available for specified tickers")

    concat = ConcatDataset(datasets)
    loader = DataLoader(concat, batch_size=batch_size, shuffle=True, collate_fn=PriceHistoryDataset.collate_fn)
    return loader


def train(
    loader,
    input_dim: int,
    emb_dim: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    from src.models.AutoEncoder import AutoEncoder

    model = AutoEncoder(input_dim=input_dim, emb_dim=emb_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    epoch_mses = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_mse = 0.0

        for src, trg in loader:
            # src, trg: (seq_len, batch)
            src = src.to(device)
            trg = trg.to(device)
            batch_size = src.size(1)

            optimizer.zero_grad()

            # Encode
            hidden, cell = model.encoder(src)

            loss = 0.0
            mse_accum = 0.0

            # Decode step-by-step: feed decoder with src[t] and predict trg[t]
            for t in range(src.size(0)):
                dec_input = src[t]  # (batch,)
                output, hidden, cell = model.decoder(dec_input, hidden, cell)  # (batch, output_dim)

                # CrossEntropy expects (batch, C) and target (batch,)
                loss_t = criterion(output, trg[t])
                loss = loss + loss_t

                # MSE between softmax probabilities and one-hot target
                probs = torch.softmax(output, dim=-1)
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, trg[t].unsqueeze(1), 1.0)
                mse_t = torch.mean((probs - one_hot) ** 2).item()
                mse_accum += mse_t

            loss = loss / src.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_accum / src.size(0)
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        avg_mse = total_mse / max(1, total_batches)
        epoch_losses.append(avg_loss)
        epoch_mses.append(avg_mse)

        print(f"Epoch {ep}/{epochs} - CE Loss: {avg_loss:.4f} - MSE(prob,onehot): {avg_mse:.6f}")

    return model, epoch_losses, epoch_mses


def main():
    ensure_paths()

    tickers = ["XLK", "XLF", "XLV", "XLY", "XLE"]
    seq_len = 32
    batch_size = 64
    num_bins = 256
    epochs = 6

    print("Building dataloader...")
    loader = build_dataloader(tickers, seq_len=seq_len, batch_size=batch_size, num_bins=num_bins)
    print("Dataloader ready. Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses, mses = train(loader, input_dim=num_bins, emb_dim=128, hidden_dim=256, n_layers=2, dropout=0.1, epochs=epochs, lr=1e-3, device=device)

    # Ensure models dir exists
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "autoencoder.pt"
    torch.save(model.state_dict(), str(model_path))
    print(f"Saved model to {model_path}")

    # Plot MSE curve
    plt.figure()
    plt.plot(range(1, len(mses) + 1), mses, marker="o")
    plt.title("MSE (probs vs one-hot) during training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plot_path = models_dir / "mse_training.png"
    plt.savefig(plot_path)
    print(f"Saved MSE plot to {plot_path}")


if __name__ == "__main__":
    main()
