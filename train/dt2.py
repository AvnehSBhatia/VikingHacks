import torch
import sys

# Patch transformers BEFORE vec2text imports it
try:
    from transformers.integrations import accelerate
    original_check = accelerate.check_and_set_device_map
    
    def patched_check(device_map):
        """Skip the meta device check that's causing issues"""
        if device_map is None:
            return None
        if isinstance(device_map, str) and device_map == "auto":
            # Don't use device_map auto, just return None
            return None
        return original_check(device_map)
    
    accelerate.check_and_set_device_map = patched_check
except ImportError:
    # In older versions of transformers, accelerate integration may not exist
    pass

import vec2text
from sentence_transformers import SentenceTransformer

print(f"--- 🍎 Apple Silicon vec2text Mode ---")

# 1. LOAD THE VEC2TEXT CORRECTOR
# This handles the iterative correction (guessing and refining)
corrector = vec2text.load_pretrained_corrector("gtr-base")

# 2. LOAD THE GTR-BASE ENCODER
# This is the model that generates the vector you want to invert
encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')

def run_inversion(input_text):
    print(f"\n[Original]: {input_text}")
    
    # Step A: Create the vector
    # SentenceTransformer's encode() normally scales and normalizes its embeddings, 
    # but the vec2text GTR-base corrector was specifically trained on unnormalized, 
    # mean-pooled T5 encoder outputs (bypassing the custom pooling/dense layer).
    # To get the right embedding to invert, we use the method provided by vec2text:
    inputs = corrector.embedder_tokenizer(
        [input_text], return_tensors="pt", max_length=128, truncation=True, padding="max_length"
    )
    
    with torch.no_grad():
        vector = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids.to(corrector.model.device),
            attention_mask=inputs.attention_mask.to(corrector.model.device),
        )
    
    # Step B: Invert the vector back to text
    # num_steps=20 is the sweet spot for GTR-Base
    with torch.inference_mode():
        result = vec2text.invert_embeddings(
            embeddings=vector,
            corrector=corrector,
            num_steps=20 
        )
    
    return result[0]

# --- EXECUTION ---
if __name__ == "__main__":
    test_phrase = "Pictures of the man were sent to his son."
    
    try:
        recovered = run_inversion(test_phrase)
        print(f"[Recovered]: {recovered}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTip: Check that you have sufficient disk space for model downloads.")