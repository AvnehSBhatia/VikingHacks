"""Project entry point; implementation is in `cohesive.models.compressor`."""

from cohesive.models.compressor import (
    BART_EMBED_DIM,
    BGE_DIM,
    CompressedUnit,
    Compressor,
    bart_decode_from_vector,
    bart_embed_text,
    bart_embed_texts,
    build_compressed_unit,
    compress,
    compressed_unit_from_paragraph,
)

__all__ = [
    "BART_EMBED_DIM",
    "BGE_DIM",
    "CompressedUnit",
    "Compressor",
    "bart_decode_from_vector",
    "bart_embed_text",
    "bart_embed_texts",
    "build_compressed_unit",
    "compress",
    "compressed_unit_from_paragraph",
]
