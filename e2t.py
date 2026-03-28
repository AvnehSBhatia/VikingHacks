import torch

# Patch transformers BEFORE vec2text imports it
try:
    from transformers.integrations import accelerate
    original_check = accelerate.check_and_set_device_map
    
    def patched_check(device_map):
        if device_map is None:
            return None
        if isinstance(device_map, str) and device_map == "auto":
            return None
        return original_check(device_map)
    
    accelerate.check_and_set_device_map = patched_check
except ImportError:
    pass

import vec2text

# Cache the corrector so it doesn't reload on every function call
_CORRECTOR = None

def get_corrector():
    global _CORRECTOR
    if _CORRECTOR is None:
        _CORRECTOR = vec2text.load_pretrained_corrector("gtr-base")
    return _CORRECTOR

def embedding_to_text(embedding: torch.Tensor, num_steps: int = 20) -> str:
    """
    Converts a given vector embedding back into text using vec2text.
    
    Args:
        embedding (torch.Tensor): The embedding tensor to invert.
        num_steps (int): The number of inversion steps (20 is standard for GTR-Base).
        
    Returns:
        str: The recovered text string.
    """
    corrector = get_corrector()
    
    # Ensure embedding has a batch dimension [1, hidden_size]
    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
        
    with torch.inference_mode():
        result = vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=corrector,
            num_steps=num_steps
        )
    
    return result[0]
