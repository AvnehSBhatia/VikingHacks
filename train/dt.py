import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

# --- CONFIG ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "sentence-transformers/gtr-t5-large"
# We need more data to train a Deep Bridge (MLP)
NUM_SAMPLES = 3000 
BATCH_SIZE = 16

# 1. THE DEEP BRIDGE ARCHITECTURE
class DeepBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
    def forward(self, x):
        return self.net(x)

# 2. LOAD MODELS
print(f"Loading models on {DEVICE}...")
bge = SentenceTransformer('BAAI/bge-large-en-v1.5').to(DEVICE)
t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# 3. PREP SYNTHETIC DATA
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
lines = [l.strip() for l in dataset['text'] if len(l.strip()) > 80][:NUM_SAMPLES]

# 4. TRAINING LOOP
bridge = DeepBridge().to(DEVICE)
optimizer = optim.Adam(bridge.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("Training Deep Bridge...")
for epoch in range(5):
    epoch_loss = 0
    for i in tqdm(range(0, len(lines), BATCH_SIZE), desc=f"Epoch {epoch+1}"):
        batch = lines[i : i + BATCH_SIZE]
        
        # Get Source (BGE)
        x = bge.encode(batch, convert_to_tensor=True, device=DEVICE)
        
        # Get Target (T5 Encoder Output)
        with torch.no_grad():
            t5_in = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            y = t5.encoder(**t5_in).last_hidden_state.mean(dim=1)
        
        # Train Step
        optimizer.zero_grad()
        pred = bridge(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Loss: {epoch_loss / (NUM_SAMPLES/BATCH_SIZE):.6f}")

# 5. INVERSION WITH DEEP BRIDGE
def invert_deep(bge_vec):
    bridge.eval()
    with torch.inference_mode():
        # Map through MLP
        translated = bridge(bge_vec.to(DEVICE))
        # Scale slightly to ensure signal strength
        translated = translated * 1.2 
        
        encoder_outputs = BaseModelOutput(last_hidden_state=translated.unsqueeze(1))
        
        output_ids = t5.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=t5.config.decoder_start_token_id,
            max_new_tokens=60,
            min_new_tokens=5,
            num_beams=5,
            repetition_penalty=2.0
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- TEST ---
test_txt = "The technology sector is facing a new wave of innovation."
vec = bge.encode([test_txt], convert_to_tensor=True)
print(f"\n[Recovered]: {invert_deep(vec)}")