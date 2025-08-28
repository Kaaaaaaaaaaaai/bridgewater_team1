import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PriceHistoryDataset(Dataset):
    """PyTorch Dataset that fetches price history and produces tokenized sequences.

    This dataset downloads price history for a symbol (via `src.data.PriceHistory.PriceHistory`),
    computes returns (or uses raw prices), discretizes values into integer tokens and
    exposes sliding-window (src, trg) pairs suitable for training sequence models.

    The dataset returns LongTensors of shape (seq_len,) for both `src` and `trg`.
    A helper `collate_fn` is provided to convert a batch into tensors of shape
    (seq_len, batch) which is the expected sequence-first layout for the RNNs
    defined under `src.models`.

    Notes / assumptions:
    - We discretize continuous price/return values into `num_bins` categories so
      the existing `Encoder`/`Decoder` which use `nn.Embedding` can consume them.
    - `trg` is the source sequence shifted by one timestep (teacher forcing layout).
    """

    def __init__(
        self,
        symbol: str,
        seq_len: int = 32,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "max",
        price_col: str = "Adj Close",
        transform: str = "returns",
        num_bins: int = 256,
    ) -> None:
        self.symbol = symbol
        self.seq_len = int(seq_len)
        self.start = start
        self.end = end
        self.period = period
        self.price_col = price_col
        self.transform = transform
        self.num_bins = int(num_bins)

        # load and prepare data
        values = self._download_series()
        if len(values) < self.seq_len + 1:
            raise ValueError(f"Not enough data for seq_len={self.seq_len}: got {len(values)} samples")

        self._fit_bins(values)
        tokens = self._tokenize(values)

        # build sliding windows (src, trg)
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(tokens) - self.seq_len):
            src = tokens[i : i + self.seq_len]
            trg = tokens[i : i + self.seq_len]
            self.pairs.append((src, trg))

    def _download_series(self) -> np.ndarray:
        # import PriceHistory lazily so this module can be imported even if yfinance
        # or the data module has extra runtime requirements.
        try:
            from src.data.PriceHistory import PriceHistory
        except Exception as e:
            raise ImportError(
                "Could not import PriceHistory. Ensure src/data/PriceHistory.py is available and its dependencies are installed."
            ) from e

        ph = PriceHistory()
        ph.symbol = self.symbol
        df = ph.get_price_history(start=self.start, end=self.end, period=self.period)
        # Accept alternate commonly-used column names (Adj Close, Close)
        if self.price_col not in df.columns:
            alt_cols = ["Adj Close", "Close"]
            found = None
            for c in [self.price_col] + alt_cols:
                if c in df.columns:
                    found = c
                    break
            if not found:
                raise KeyError(f"Price column '{self.price_col}' not found in downloaded data")
            if found != self.price_col:
                print(f"Using fallback price column '{found}' for symbol {self.symbol}")
            series = df[found].dropna()
        else:
            series = df[self.price_col].dropna()
        if self.transform == "returns":
            series = series.pct_change().dropna()
        # convert to numpy float array
        return series.to_numpy(dtype=float)

    def _fit_bins(self, values: np.ndarray) -> None:
        # compute bin edges based on observed min/max using linear bins
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
            # fallback to small symmetric range around zero
            vmin, vmax = -1.0, 1.0
        self.bin_edges = np.linspace(vmin, vmax, self.num_bins + 1)

    def _tokenize(self, values: np.ndarray) -> np.ndarray:
        # map continuous values to integer tokens in [0, num_bins-1]
        toks = np.digitize(values, self.bin_edges) - 1
        # clip to valid range
        toks = np.clip(toks, 0, self.num_bins - 1).astype(np.int64)
        return toks

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        src, trg = self.pairs[idx]
        return torch.from_numpy(src).long(), torch.from_numpy(trg).long()

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Collate a list of (src, trg) pairs into tensors shaped (seq_len, batch).

        Returns:
            src: LongTensor of shape (seq_len, batch)
            trg: LongTensor of shape (seq_len, batch)
        """
        srcs = torch.stack([b[0] for b in batch], dim=0)  # (batch, seq_len)
        trgs = torch.stack([b[1] for b in batch], dim=0)  # (batch, seq_len)
        # transpose to (seq_len, batch) to match encoder/decoder expectations
        return srcs.transpose(0, 1).contiguous(), trgs.transpose(0, 1).contiguous()
