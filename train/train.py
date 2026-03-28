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

# --- CONFIG ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BRIDGE_SAVE_PATH = "bge_to_t5_bridge.joblib"
MODEL_NAME = "sentence-transformers/gtr-t5-large"
NUM_SAMPLES = 1500  # More samples = better "accent"
BATCH_SIZE = 16

print(f"--- Apple Silicon Recovery Mode ({DEVICE}) ---")

# 1. MODELS
bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5').to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32
).to(DEVICE)
t5_model.eval()

# 2. BRIDGE
def get_or_build_bridge():
    if os.path.exists(BRIDGE_SAVE_PATH):
        return joblib.load(BRIDGE_SAVE_PATH)

    print(f"Calibrating Bridge...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    lines = [line.strip() for line in dataset['text'] if len(line.strip()) > 80][:NUM_SAMPLES]

    x_bge = bge_model.encode(lines, batch_size=BATCH_SIZE, show_progress_bar=True)
    y_t5_list = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), BATCH_SIZE), desc="T5 Encoding"):
            batch = lines[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            outputs = t5_model.encoder(**inputs).last_hidden_state
            y_t5_list.append(outputs.mean(dim=1).cpu().numpy())
            
    y_t5 = np.vstack(y_t5_list)
    regr = Ridge(alpha=0.1) # Lower alpha for a tighter fit
    regr.fit(x_bge, y_t5)
    joblib.dump(regr, BRIDGE_SAVE_PATH)
    return regr

mapper = get_or_build_bridge()

# 3. INVERSION
def invert_embedding(bge_vector):
    if torch.is_tensor(bge_vector):
        bge_vector = bge_vector.detach().cpu().numpy()
    
    # A. Map and Scale
    translated = mapper.predict(bge_vector.reshape(1, -1))
    # We scale by 10 to wake up the T5 attention layers
    translated_tensor = torch.from_numpy(translated).to(DEVICE) * 10.0
    
    encoder_outputs = BaseModelOutput(last_hidden_state=translated_tensor.unsqueeze(1))
    start_id = t5_model.config.decoder_start_token_id or tokenizer.pad_token_id

    # B. Forced Generation
    with torch.inference_mode():
        output_ids = t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=start_id,
            max_new_tokens=100,
            min_new_tokens=10,       # FORCE IT TO TALK
            num_beams=5,
            no_repeat_ngram_size=3,  # Prevent word loops
            repetition_penalty=2.5,  # Punish repetitive garbage
            length_penalty=1.5,      # Reward longer sentences
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_phrase = "The artificial intelligence system decoded the hidden message successfully."
    print(f"\n[Original]: {test_phrase}")
    
    vec = bge_model.encode([test_phrase])
    result = invert_embedding(vec)
    
    print(f"[Recovered]: {result}")