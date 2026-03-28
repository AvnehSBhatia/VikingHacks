import torch
import numpy as np
import os
import joblib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.linear_model import Ridge

# --- APPLE SILICON CONFIG ---
# Use 'mps' for Metal acceleration on Mac
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BRIDGE_SAVE_PATH = "bge_to_t5_bridge.joblib"
MODEL_NAME = "sentence-transformers/gtr-t5-large"
NUM_SAMPLES = 2000  # Mac can handle a bit more data easily
BATCH_SIZE = 16     # Unified memory allows larger batches than WSL

print(f"--- Initializing Models on {DEVICE} ---")

# 1. Load BGE
bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5').to(DEVICE)

# 2. Load T5 with MPS-friendly settings
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, # MPS prefers float32 for certain T5 operations to avoid 'NaN'
    low_cpu_mem_usage=True
).to(DEVICE)
t5_model.eval()

# --- BRIDGE BUILDING ---
def get_or_build_bridge():
    if os.path.exists(BRIDGE_SAVE_PATH):
        print(f"Loading existing bridge from {BRIDGE_SAVE_PATH}...")
        return joblib.load(BRIDGE_SAVE_PATH)

    print(f"No bridge found. Calibrating {NUM_SAMPLES} samples on Metal GPU...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    lines = [line.strip() for line in dataset['text'] if len(line.strip()) > 60][:NUM_SAMPLES]

    # Step 1: BGE Source Vectors
    print("Step 1/3: Encoding BGE Vectors...")
    x_bge = bge_model.encode(lines, batch_size=BATCH_SIZE, show_progress_bar=True)

    # Step 2: T5 Target Vectors
    print("Step 2/3: Encoding T5 Target Space...")
    y_t5_list = []
    with torch.no_grad():
        for i in range(0, len(lines), BATCH_SIZE):
            batch = lines[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            
            # MPS specific: Ensure inputs are in the right format
            outputs = t5_model.encoder(**inputs).last_hidden_state
            
            # Mean pool and move to CPU for Scikit-Learn
            pooled = outputs.mean(dim=1).cpu().numpy()
            y_t5_list.append(pooled)
            
    y_t5 = np.vstack(y_t5_list)

    # Step 3: Fit Mapper (CPU handles this faster than GPU for Ridge)
    print("Step 3/3: Fitting Ridge Regression...")
    regr = Ridge(alpha=1.0)
    regr.fit(x_bge, y_t5)
    
    joblib.dump(regr, BRIDGE_SAVE_PATH)
    print(f"Bridge saved to {BRIDGE_SAVE_PATH}")
    return regr

mapper = get_or_build_bridge()

# --- THE INVERSION FUNCTION ---
def invert_embedding(bge_vector):
    if torch.is_tensor(bge_vector):
        bge_vector = bge_vector.detach().cpu().numpy()
    if len(bge_vector.shape) == 1:
        bge_vector = bge_vector.reshape(1, -1)

    # A. Map Space
    translated = mapper.predict(bge_vector)
    translated_tensor = torch.from_numpy(translated).to(DEVICE)
    
    # B. Decoder Prep
    translated_tensor = translated_tensor.unsqueeze(1) 
    encoder_outputs = BaseModelOutput(last_hidden_state=translated_tensor)
    start_id = t5_model.config.decoder_start_token_id or tokenizer.pad_token_id

    # C. Generate
    with torch.no_grad():
        output_ids = t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=start_id,
            max_new_tokens=64,
            num_beams=5,
            repetition_penalty=1.5,
            early_stopping=True
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_phrase = "The silicon chips are optimized for machine learning tasks."
    print(f"\n[Input]: {test_phrase}")

    mystery_vector = bge_model.encode([test_phrase])
    result = invert_embedding(mystery_vector)
    
    print(f"[Recovered]: {result}")