"""
Train v4 hallucination corrector:

  Phase 1 — contrastive (frozen GTR + temp projection head):
            anchor=dialogue, positive=dialogue+right, negative=dialogue+hallu.
            GTR stays frozen; a linear projection head is trained and discarded.

  Phase 2 — freeze encoder, train learnable S + 4-block residual:
            All BART summaries + GTR embeddings are pre-computed once, then
            corrector(z_sum) ≈ z_gt is trained over N epochs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

_pkg = Path(__file__).resolve().parent.parent  # cohesive/
_ROOT = Path(__file__).resolve().parents[2]  # repo root
if not __package__:
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from models.hallu_corrector_module import HalluCorrectorModule
    from models.hallucination_latent import HalluCorrectorPipeline
    from models.sentence_encoder import GTRSentenceEncoder
else:
    from ..models.hallu_corrector_module import HalluCorrectorModule
    from ..models.hallucination_latent import HalluCorrectorPipeline
    from ..models.sentence_encoder import GTRSentenceEncoder

try:
    from ..data.dataloader import load_training_dataframe
except ImportError:
    from cohesive.data.dataloader import load_training_dataframe

try:
    from ..constants import EMBED_DIM
except ImportError:
    from cohesive.constants import EMBED_DIM


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """2-class InfoNCE: anchor closer to positive than negative."""
    pos_sim = (anchor * positive).sum(dim=-1) / temperature
    neg_sim = (anchor * negative).sum(dim=-1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


class TripleDataset(Dataset):
    def __init__(self, df):
        self.rows = [
            (
                str(r["dialogue_history"]),
                str(r["right_response"]),
                str(r["hallucinated_response"]),
            )
            for _, r in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        d, r, h = self.rows[i]
        return {
            "dialogue": d,
            "right": r,
            "hallu": h,
            "anchor": d,
            "positive": f"{d}\n{r}",
            "negative": f"{d}\n{h}",
        }


def collate(batch):
    keys = batch[0].keys()
    return {k: [b[k] for b in batch] for k in keys}


def train_phase_contrastive(
    encoder: GTRSentenceEncoder,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> None:
    """
    Contrastive warmup with a frozen GTR encoder.

    A small linear projection head is trained on frozen GTR embeddings and
    discarded afterward — avoids backpropping through T5 (NaN on MPS).
    """
    dev = torch.device(device)
    proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False).to(dev)
    nn.init.eye_(proj.weight)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    encoder.eval()

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total = 0.0
        n_b = 0
        pbar = tqdm(loader, desc=f"[Contrastive] epoch {ep}/{epochs}", leave=False)
        for batch in pbar:
            opt.zero_grad()
            with torch.no_grad():
                a = encoder.encode_texts(batch["anchor"]).to(dev)
                p = encoder.encode_texts(batch["positive"]).to(dev)
                n_ = encoder.encode_texts(batch["negative"]).to(dev)
            a_p = F.normalize(proj(a), dim=-1)
            p_p = F.normalize(proj(p), dim=-1)
            n_p = F.normalize(proj(n_), dim=-1)
            loss = info_nce_loss(a_p, p_p, n_p)
            if not torch.isfinite(loss):
                pbar.set_postfix(loss="NaN—skipped")
                continue
            loss.backward()
            opt.step()
            total += loss.item()
            n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(
            f"[Contrastive] epoch {ep}/{epochs}  "
            f"loss={total/max(n_b, 1):.4f}  "
            f"time={time.perf_counter()-t0:.1f}s"
        )
    # proj discarded — encoder unchanged


def _precompute_corrector_pairs(
    encoder: GTRSentenceEncoder,
    rows: list[tuple[str, str, str]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute (z_sum, z_gt) for every training row (runs once before Phase 2).

    Calling BART + GTR embed per-row inside the training loop would mean
    N_rows × epochs BART calls. Pre-computing reduces this to N_rows calls.

    Returns z_sum_all [N, D] and z_gt_all [N, D] on CPU.
    """
    try:
        from cohesive.summarizer import summarize_hallucination_branch
    except ImportError:
        from summarizer import summarize_hallucination_branch  # type: ignore

    dev = torch.device(device)
    z_sums, z_gts = [], []

    encoder.eval()
    print(f"[Phase 2 pre-compute] summarising + embedding {len(rows)} rows…")
    for dlg, r, h in tqdm(rows, desc="  BART+GTR pre-compute"):
        summ = summarize_hallucination_branch(dlg, h, device=device)
        with torch.no_grad():
            z_sum = encoder.encode_texts([summ]).squeeze(0).to(dev)
            z_gt = encoder.encode_texts([f"{dlg}\n{r}"]).squeeze(0).to(dev)
        z_sums.append(z_sum.cpu())
        z_gts.append(z_gt.cpu())

    return torch.stack(z_sums), torch.stack(z_gts)


def train_phase_corrector(
    encoder: GTRSentenceEncoder,
    corrector: HalluCorrectorModule,
    rows: list[tuple[str, str, str]],
    device: str,
    epochs: int,
    lr: float,
    batch_size: int = 8,
) -> None:
    dev = torch.device(device)

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Pre-compute all summaries + embeddings once
    z_sum_all, z_gt_all = _precompute_corrector_pairs(encoder, rows, device)

    ds = TensorDataset(z_sum_all, z_gt_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        drop_last=len(ds) >= batch_size)

    opt = torch.optim.AdamW(corrector.parameters(), lr=lr, weight_decay=1e-4)
    corrector.train()

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total = 0.0
        n_b = 0
        pbar = tqdm(loader, desc=f"[Corrector]  epoch {ep}/{epochs}", leave=False)
        for z_sum_b, z_gt_b in pbar:
            z_sum_b = z_sum_b.to(dev)
            z_gt_b = z_gt_b.to(dev)
            opt.zero_grad()
            z_hat = corrector(z_sum_b)
            cos = (z_hat * z_gt_b).sum(dim=-1)
            loss = (1.0 - cos).mean()
            if not torch.isfinite(loss):
                pbar.set_postfix(loss="NaN—skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(
            f"[Corrector]  epoch {ep}/{epochs}  "
            f"L1-cos={total/max(n_b, 1):.4f}  "
            f"time={time.perf_counter()-t0:.1f}s"
        )


def train(
    out_dir: str,
    dataframe,
    *,
    device: str | None = None,
    batch_size: int = 8,
    contrastive_epochs: int = 5,
    corrector_epochs: int = 15,
    lr_contrastive: float = 1e-3,
    lr_corrector: float = 1e-3,
    max_rows: int | None = None,
):
    dev = device or _default_device()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if max_rows is not None:
        dataframe = dataframe.head(max_rows)

    print(f"[Train] device={dev}  rows={len(dataframe)}")
    ds = TripleDataset(dataframe)
    n = len(ds)
    if n < 2:
        raise ValueError("Need at least 2 rows")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=n >= batch_size,
    )

    encoder = GTRSentenceEncoder(device=dev)
    encoder.to(dev)

    print("[Train] Phase 1 — contrastive (frozen GTR + projection head)")
    train_phase_contrastive(
        encoder, loader, dev, epochs=contrastive_epochs, lr=lr_contrastive
    )

    corrector = HalluCorrectorModule(EMBED_DIM).to(dev)
    rows = ds.rows
    print("[Train] Phase 2 — S + residual corrector")
    train_phase_corrector(
        encoder, corrector, rows, dev,
        epochs=corrector_epochs,
        lr=lr_corrector,
        batch_size=batch_size,
    )

    pipe = HalluCorrectorPipeline(encoder, corrector, device=dev)
    path = os.path.join(out_dir, "best.pt")
    pipe.save(path)
    print(f"[Train] Done → {path}")


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from cohesive.data.dataloader import load_training_dataframe

    train(
        out_dir=str(_ROOT / "checkpoints"),
        dataframe=load_training_dataframe(max_rows=500),
        batch_size=4,
        contrastive_epochs=7,
        corrector_epochs=10,
        max_rows=500,
    )
