import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

# --- CONFIG ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "sentence-transformers/gtr-t5-large"
CHECKPOINT_PATH = "bridge_v3_checkpoint.pth"
NUM_SAMPLES = 2500 
BATCH_SIZE = 16

print(f"--- 🍎 Apple Silicon Forced Generation ({DEVICE}) ---")

# 1. THE ARCHITECTURE
class DeepBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
    def forward(self, x):
        return self.net(x)

# 2. LOAD MODELS
bge = SentenceTransformer('BAAI/bge-large-en-v1.5').to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
t5.eval()

bridge = DeepBridge().to(DEVICE)

# 3. THE CHECKPOINT & TRAINING LOGIC
def train_and_save():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Bridge file found! Loading weights from {CHECKPOINT_PATH}...")
        bridge.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        return

    print(f"🚀 No bridge found. Training on {NUM_SAMPLES} samples...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    lines = [l.strip() for l in dataset['text'] if len(l.strip()) > 80][:NUM_SAMPLES]

    optimizer = optim.Adam(bridge.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(2):
        epoch_loss = 0
        for i in tqdm(range(0, len(lines), BATCH_SIZE), desc=f"Epoch {epoch+1}"):
            batch = lines[i : i + BATCH_SIZE]
            x_vecs = bge.encode(batch, convert_to_tensor=True, device=DEVICE)
            
            with torch.inference_mode():
                t5_in = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                y_vecs = t5.encoder(**t5_in).last_hidden_state.mean(dim=1)

            optimizer.zero_grad()
            pred = bridge(x_vecs)
            loss = criterion(pred, y_vecs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Checkpoint save
        torch.save(bridge.state_dict(), CHECKPOINT_PATH)
        print(f"💾 Epoch {epoch+1} saved to {CHECKPOINT_PATH}")

# 4. THE FORCED INVERSION FUNCTION
def invert_max_volume(bge_vec):
    bridge.eval()
    with torch.inference_mode():
        # Map through MLP and add a 2x Gain to wake up the T5 decoder
        translated = bridge(bge_vec.to(DEVICE)) * 2.0 
        
        encoder_outputs = BaseModelOutput(last_hidden_state=translated.unsqueeze(1))
        
        # EXTREME GENERATION SETTINGS
        output_ids = t5.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=t5.config.decoder_start_token_id or tokenizer.pad_token_id,
            max_new_tokens=128,      # Double the limit
            min_new_tokens=25,       # FORCE IT TO TALK (At least 25 words)
            num_beams=10,            # High beam search for more paths
            length_penalty=2.0,      # Heavily reward longer outputs
            repetition_penalty=3.5,  # Drastic penalty for repeating words
            no_repeat_ngram_size=3,  # Prevents triple-word loops
            early_stopping=False,    # Do not stop until it hits max or a real EOS
            do_sample=True,          # Add randomness to escape "Empty String" traps
            temperature=0.9
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- EXECUTION ---
if __name__ == "__main__":
    train_and_save()

    test_phrase = "Global financial markets are reacting to the shift in interest rates."
    print(f"\n[Original]: {test_phrase}")
    
    mystery_vec = bge.encode([test_phrase], convert_to_tensor=True)
    result = invert_max_volume(mystery_vec)
    
    print(f"\n[Recovered (Forced)]: {result}")