import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.linear_model import Ridge # Ridge is more stable for high-dims

# 1. SETUP MODELS & DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# BGE is our source (1024-dim)
bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5').to(device)

# T5 is our decoder (1024-dim)
model_name = "sentence-transformers/gtr-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=False
).to(device)

# 2. SYNTHETIC DATA ALIGNMENT (The "Bridge")
def build_bridge(num_samples=1000):
    print(f"Downloading {num_samples} sentences for calibration...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    # Clean and filter for actual sentences
    lines = [line.strip() for line in dataset['text'] if len(line.strip()) > 60][:num_samples]
    
    print("Encoding source (BGE)...")
    x_bge = bge_model.encode(lines)
    
    print("Encoding target (T5 Latent Space)...")
    with torch.no_grad():
        inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt").to(device)
        # We want the 'thought' vector T5 produces internally
        y_t5 = t5_model.encoder(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    
    print("Calculating translation matrix...")
    # Ridge regression maps BGE space -> T5 space
    regr = Ridge(alpha=1.0)
    regr.fit(x_bge, y_t5)
    return regr

# Initialize the bridge
mapper = build_bridge(num_samples=1000)

# 3. THE FINAL INVERSION FUNCTION
def invert_embedding(bge_vector):
    """
    Takes a 1024-dim BGE vector and returns text.
    """
    # Ensure it's a numpy array for the mapper
    if torch.is_tensor(bge_vector):
        bge_vector = bge_vector.detach().cpu().numpy()
    
    if len(bge_vector.shape) == 1:
        bge_vector = bge_vector.reshape(1, -1)

    # Step A: Translate BGE math to T5 math
    translated = mapper.predict(bge_vector)
    translated_tensor = torch.from_numpy(translated).to(device, dtype=torch.float32)
    
    # Step B: Prepare for Decoder (Shape: [Batch, Seq_Len, Dim])
    translated_tensor = translated_tensor.unsqueeze(1) 
    encoder_outputs = BaseModelOutput(last_hidden_state=translated_tensor)
    
    # Step C: Generate
    start_id = t5_model.config.decoder_start_token_id or tokenizer.pad_token_id
    
    with torch.no_grad():
        output_ids = t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=start_id,
            max_new_tokens=64,
            num_beams=5,
            repetition_penalty=1.2,
            early_stopping=True
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 4. TEST IT
test_text = "The board of directors is meeting to discuss the annual budget."
print(f"\nOriginal Text: {test_text}")

# Create the mystery vector
mystery_vector = bge_model.encode([test_text])

# Invert it
recovered_text = invert_embedding(mystery_vector)
print(f"Recovered Text: {recovered_text}")