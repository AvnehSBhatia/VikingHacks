import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from sentence_transformers import SentenceTransformer

# 1. SETUP DEVICES
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. LOAD BGE ENCODER (The model that made your embedding)
encoder = SentenceTransformer('BAAI/bge-large-en-v1.5').to(device)

# 3. LOAD DECODER (Standard T5 model)
# We use a model trained to turn 1024-dim vectors into text
model_name = "sentence-transformers/gtr-t5-large" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    low_cpu_mem_usage=False, # CRITICAL: Disables the meta-device behavior
    device_map=None          # Prevents 'accelerate' from making bad device choices
).to(device)

# 4. THE INVERSION FUNCTION
def invert_embedding(embedding_vector):
    embedding_vector = embedding_vector.to(device=model.device, dtype=model.dtype)
    
    if len(embedding_vector.shape) == 1:
        embedding_vector = embedding_vector.unsqueeze(0).unsqueeze(0)
    elif len(embedding_vector.shape) == 2:
        embedding_vector = embedding_vector.unsqueeze(1)

    embedding_vector = embedding_vector.to(model.dtype)  # Ensure the embedding is in the same dtype as the model
    encoder_outputs = BaseModelOutput(last_hidden_state=embedding_vector)
    decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.pad_token_id

    with torch.no_grad():
        output_ids = model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=decoder_start_token_id,
            max_new_tokens=64,      # Ensure it tries to generate
            min_new_tokens=5,       # FORCE it to say at least a few words
            num_beams=10,           # More beams = more thorough search
            repetition_penalty=1.5, # Prevents it from getting stuck or looping
            do_sample=True,         # Adding some randomness can help "break" the silence
            temperature=0.7
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 5. TEST IT
original_text = "The artificial intelligence revolution is changing how we write code."
vector = encoder.encode([original_text], convert_to_tensor=True)

reconstructed_text = invert_embedding(vector)
print(repr(reconstructed_text))
print(f"Reconstructed: {reconstructed_text}")