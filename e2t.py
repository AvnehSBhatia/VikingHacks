import vec2text as v2t
import torch
from sentence_transformers import SentenceTransformer as ST

encoder = ST("BAAI/bge-large-en-v1.5")
decoder = v2t.load_pretrained_corrector("jxm/gtr__nq__32")
target_embeddings = torch.randn(1, 1024).cpu()

reconstructed_text = v2t.invert_embeddings(
    embeddings=target_embeddings,
    corrector=decoder,
    num_steps=20
)

print(reconstructed_text)