"""Shared dimensions and model ids for the GTR / vec2text stack."""

# Aligns with sentence-transformers/gtr-t5-base and vec2text gtr-base corrector.
EMBED_DIM = 768
GTR_MODEL_NAME = "sentence-transformers/gtr-t5-base"

# Checkpoint format after the v3 autoreg era; v4 = GTR + stretch + residual corrector.
CKPT_VERSION_V4 = 4
