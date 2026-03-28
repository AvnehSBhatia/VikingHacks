import os
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.linear_model import Ridge

# --- CONFIG (no I/O or model load at import) ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SAVE_PATH = str(_REPO_ROOT / "bge_to_t5_bridge.joblib")
MODEL_NAME = "sentence-transformers/gtr-t5-large"
NUM_SAMPLES = 1500  # More samples = better "accent"
BATCH_SIZE = 16


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class InversionPipeline:
    """BGE encoder, Ridge bridge, and T5 decoder for embedding inversion."""

    device: str
    bge_model: SentenceTransformer
    tokenizer: T5Tokenizer
    t5_model: T5ForConditionalGeneration
    mapper: Ridge


_pipeline_singleton: Optional[InversionPipeline] = None


def get_or_build_bridge(
    bge_model: SentenceTransformer,
    tokenizer: T5Tokenizer,
    t5_model: T5ForConditionalGeneration,
    device: str,
    bridge_save_path: str = BRIDGE_SAVE_PATH,
) -> Ridge:
    if os.path.exists(bridge_save_path):
        return joblib.load(bridge_save_path)

    print("Calibrating Bridge...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    lines = [line.strip() for line in dataset["text"] if len(line.strip()) > 80][:NUM_SAMPLES]

    x_bge = bge_model.encode(lines, batch_size=BATCH_SIZE, show_progress_bar=True)
    y_t5_list = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), BATCH_SIZE), desc="T5 Encoding"):
            batch = lines[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = t5_model.encoder(**inputs).last_hidden_state
            y_t5_list.append(outputs.mean(dim=1).cpu().numpy())

    y_t5 = np.vstack(y_t5_list)
    regr = Ridge(alpha=0.1)
    regr.fit(x_bge, y_t5)
    joblib.dump(regr, bridge_save_path)
    return regr


def load_pipeline(
    device: Optional[str] = None,
    bridge_save_path: str = BRIDGE_SAVE_PATH,
    build_bridge_if_missing: bool = True,
) -> InversionPipeline:
    """
    Load BGE, T5, and Ridge bridge. Safe to call multiple times; returns a fresh pipeline each time.

    If ``build_bridge_if_missing`` is False and the bridge file is absent, raises FileNotFoundError.
    """
    dev = device or _default_device()
    print(f"--- Loading inversion pipeline on {dev} ---")

    bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5").to(dev)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    t5_model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(dev)
    t5_model.eval()

    if not os.path.exists(bridge_save_path):
        if not build_bridge_if_missing:
            raise FileNotFoundError(
                f"Bridge not found at {bridge_save_path} and build_bridge_if_missing=False"
            )
        mapper = get_or_build_bridge(bge_model, tokenizer, t5_model, dev, bridge_save_path)
    else:
        mapper = joblib.load(bridge_save_path)

    return InversionPipeline(
        device=dev,
        bge_model=bge_model,
        tokenizer=tokenizer,
        t5_model=t5_model,
        mapper=mapper,
    )


def get_pipeline_lazy() -> InversionPipeline:
    """Single shared pipeline instance (lazy)."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = load_pipeline()
    return _pipeline_singleton


def invert_embedding(
    bge_vector,
    pipeline: Optional[InversionPipeline] = None,
    *,
    bridge_scale: float = 10.0,
) -> str:
    """
    Map a BGE 1024-D vector through the Ridge bridge and decode with T5.

    ``bge_vector`` may be numpy array or torch tensor shaped [1024] or [1, 1024].
    """
    pipe = pipeline or get_pipeline_lazy()
    if torch.is_tensor(bge_vector):
        bge_vector = bge_vector.detach().cpu().numpy()

    translated = pipe.mapper.predict(bge_vector.reshape(1, -1))
    translated_tensor = torch.from_numpy(translated).to(pipe.device) * bridge_scale

    encoder_outputs = BaseModelOutput(last_hidden_state=translated_tensor.unsqueeze(1))
    start_id = pipe.t5_model.config.decoder_start_token_id or pipe.tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = pipe.t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=start_id,
            max_new_tokens=100,
            min_new_tokens=10,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.5,
            length_penalty=1.5,
            early_stopping=True,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )

    return pipe.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    test_phrase = "The artificial intelligence system decoded the hidden message successfully."
    print(f"\n[Original]: {test_phrase}")

    pl = load_pipeline()
    vec = pl.bge_model.encode([test_phrase])
    result = invert_embedding(vec, pipeline=pl)

    print(f"[Recovered]: {result}")
