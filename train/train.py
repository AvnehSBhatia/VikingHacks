import torch
import numpy as np
import os
import joblib
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.linear_model import Ridge

# --- CONFIGURATION ---
# Use 'mps' for Apple Silicon Metal acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BRIDGE_SAVE_PATH = "bge_to_t5_bridge.joblib"
MODEL_NAME = "sentence-transformers/gtr-t5-large" # 1024-dim
NUM_SAMPLES = 1200  # Balanced for speed and accuracy
BATCH_SIZE = 16     # Optimized for Mac Unified Memory

print(f"--- Running on Apple Silicon GPU ({DEVICE}) ---")

# 1. LOAD MODELS
# BGE-Large (Source)
bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5').to(DEVICE)

# T5-Large (Decoder)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, # float32 is more stable on MPS for T5
    low_cpu_mem_usage=True
).to(DEVICE)
t5_model.eval()

# --- THE BRIDGE BUILDER ---
def get_or_build_bridge():
    if os.path.exists(BRIDGE_SAVE_PATH):
        print(f"✅ Loading existing bridge from {BRIDGE_SAVE_PATH}...")
        return joblib.load(BRIDGE_SAVE_PATH)

    print(f"🚀 No bridge found. Calibrating {NUM_SAMPLES} samples (One-time process)...")
    
    # Load a small slice of Wikipedia for synthetic data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    lines = [line.strip() for line in dataset['text'] if len(line.strip()) > 70][:NUM_SAMPLES]

    # Step 1: Encode BGE (Source)
    print("\nStep 1/3: Encoding BGE Space...")
    x_bge = bge_model.encode(lines, batch_size=BATCH_SIZE, show_progress_bar=True)

    # Step 2: Encode T5 (Target) with Progress Bar
    print("\nStep 2/3: Encoding T5 Latent Space (This takes a moment)...")
    y_t5_list = []
    
    # We use inference_mode for better speed on Mac
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), BATCH_SIZE), desc="T5 Encoding Batches"):
            batch = lines[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            
            # Extract the 'thought' vector from the T5 encoder
            outputs = t5_model.encoder(**inputs).last_hidden_state
            
            # Mean pooling: [Batch, Tokens, 1024] -> [Batch, 1024]
            # This prevents the 15GB memory buffer error
            pooled = outputs.mean(dim=1).cpu().numpy()
            y_t5_list.append(pooled)
            
            # Keep Metal memory clean
            if DEVICE == "mps":
                torch.mps.empty_cache()

    y_t5 = np.vstack(y_t5_list)

    # Step 3: Solve the Map (Ridge Regression)
    print("\nStep 3/3: Fitting Linear Mapper (CPU)...")
    regr = Ridge(alpha=1.0)
    regr.fit(x_bge, y_t5)
    
    joblib.dump(regr, BRIDGE_SAVE_PATH)
    print(f"Done! Bridge saved to {BRIDGE_SAVE_PATH}")
    return regr

# Initialize or Load
mapper = get_or_build_bridge()

# --- THE INVERSION FUNCTION ---
def invert_embedding(bge_vector):
    """
    Reverse-engineers a BGE vector back into English text.
    """
    if torch.is_tensor(bge_vector):
        bge_vector = bge_vector.detach().cpu().numpy()
    if len(bge_vector.shape) == 1:
        bge_vector = bge_vector.reshape(1, -1)

    # A. Translate BGE math to T5 math using our Bridge
    translated = mapper.predict(bge_vector)
    translated_tensor = torch.from_numpy(translated).to(DEVICE)
    
    # B. Inject into the T5 Decoder pipeline
    translated_tensor = translated_tensor.unsqueeze(1) 
    encoder_outputs = BaseModelOutput(last_hidden_state=translated_tensor)
    start_id = t5_model.config.decoder_start_token_id or tokenizer.pad_token_id

    # C. Autoregressive Generation
    with torch.inference_mode():
        output_ids = t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=start_id,
            max_new_tokens=80,
            num_beams=5,
            repetition_penalty=1.5,
            early_stopping=True
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- RUN TEST ---
if __name__ == "__main__":
    # Feel free to change this to any text you want to test
    test_phrase = "Quantum computing relies on qubits to perform complex calculations."
    print(f"\n[Original]: {test_phrase}")

    # 1. Turn text into a vector
    mystery_vector = bge_model.encode([test_phrase])

    # 2. Turn vector back into text
    result = invert_embedding(mystery_vector)
    print(f"[Recovered]: {result}")