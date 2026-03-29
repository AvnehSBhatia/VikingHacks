# MemGuard — Memory layer for long-running agents

**MemGuard** is a memory layer that sits between raw conversation history and an LLM's context window. It **compresses** what happened, **eliminates** what drifted from the facts, and **reconstructs** a clean payload before the next prompt is sent.

The idea: long-running agents slowly degrade when their "memory" is naive truncation, lossy summarization, or unchecked hallucination. MemGuard avoids blind summarization by maintaining a continuously updated embedding space where sentence-level vectors are shaped by ground truth and recent context. It owns the path from **raw history -> injected context** and treats each transformation as an auditable step.

---

## End-to-end pipeline

```
Raw history → Compression → Elimination → Reconstruction → LLM context
```

| Stage | Role | What it solves |
|--------|------|----------------|
| **Compression** | Turn many messages into one fluent paragraph that still reflects structure | Raw logs are too long and noisy for the window |
| **Elimination** | Strip claims the summary cannot justify against originals | Summaries drift; drift compounds |
| **Reconstruction** | Assemble only verified material into final context | You inject truth-shaped text, not another vague summary |

---

## Stage 1 — Compression

The compressor (`compressor.py`, `compress(...)`) turns a list of messages into **one fluent paragraph**. It is **not** “keep the last *k* messages.” It scores local importance, selects a diverse subset under a **BART token budget**, then summarizes.

### Step 1 — Sliding window scoring

Each message is embedded with **sentence-transformers** using **`all-mpnet-base-v2`**. For message index `i`, the score is the **average cosine similarity** between that embedding and embeddings of neighbors in a **window of size 2** on each side: indices roughly `i-2` through `i+2`, excluding `i` itself and out-of-range indices.

**Why it matters:** importance is **local**. A line that only makes sense next to its setup (or punchline) gets a low score when isolated from that neighborhood. Coherent exchanges tend to score higher together.

Cosine similarity is implemented manually (e.g. dot product divided by the product of L2 norms), consistent with the rest of the stack.

### Step 2 — MMR selection (with a token ceiling)

**Maximal Marginal Relevance (MMR)** is implemented from scratch with **NumPy** (no MMR library). At each step, every remaining candidate gets a score:

\[
\text{MMR} = \lambda \cdot \text{relevance} - (1 - \lambda) \cdot \max\_{\text{selected}} \text{cosine\_sim(candidate, selected)}
\]

- **Relevance** = sliding-window score from Step 1.  
- **Diversity** = how different the candidate is from everything already chosen (max similarity to selected set).  
- **`lambda_mmr`** (default `0.5`) trades off relevance vs diversity.

**Stopping rule (implemented):** the loop does **not** use a fixed “pick *n* messages.” It adds the best MMR candidate only if the **role-prefixed string** that would be sent to BART stays at or under **`max_tokens`** (default **800**), as measured by the **same tokenizer** as `facebook/bart-large-cnn`. If the next best pick would exceed the budget, selection stops. The first pick is still allowed even in pathological cases so you never return an empty selection from an empty budget alone. Selected indices are **re-sorted by original index** so the summarizer sees **true conversation order**.

**Effect:** the compressor is **self-regulating** with respect to BART’s input limit—no separate manual `target_count` tuning for typical use.

### Step 3 — BART summarization

MMR-selected messages are formatted as a single string with **role prefixes** (e.g. `User: ... Assistant: ...`) and passed through **`facebook/bart-large-cnn`** via the **Hugging Face `transformers` pipeline** (with fallbacks across `transformers` versions where task names differ). The returned summary string is the **compressed paragraph** passed to elimination.

**Local inference:** compression runs in-process; models load once at module import. Optional **Hugging Face token** can be supplied via environment (e.g. `HF_TOKEN` in `.env`, loaded with `python-dotenv`) for gated downloads.

---

## Stage 2 — Elimination

The **eliminator** takes the **compressed paragraph** from Stage 1 and checks it against the **original raw messages** — the immutable ground truth.

**Intended behavior:**

1. **Claims** (or atomic propositions) are extracted from the paragraph.  
2. Each claim is **embedded** with the same embedding stack as compression (**`all-mpnet-base-v2`**).  
3. Claims are compared to the **source messages** they should reflect, using **cosine similarity** (same manual definition as elsewhere).  
4. Anything that **does not align** with the originals above a configurable threshold is **removed**, not silently rewritten.  
5. **Audit trail:** flagged items can be logged with similarity scores, which source span was used, and why the line was dropped.

**Principle:** the raw transcript is fixed. The summary either **faithfully represents** it or it doesn’t — that’s a **measurable** check. Lower similarity thresholds → more aggressive stripping; higher thresholds → more lenient.

*(Implementation of the eliminator may live in a separate module; the contract above is the product design.)*

---

## Stage 3 — Reconstruction

The **reconstructor** receives **only** the claims (or fragments) that **survived** elimination.

**Intended behavior:**

- This is **not** another open-ended summarization pass whose job is to “be creative.”  
- It is **structured assembly**: ordering and light glue text so the result reads as **one coherent context block** for the next turn.  
- Output = the **context window payload**: compact, accurate, and **audited** in the sense that every piece passed elimination.

Together, elimination + reconstruction implement **MemGuard-style** discipline: compress for density, then **cut** drift, then **rebuild** only what you can defend.

---

## Tech stack (compression + shared primitives)

| Piece | Role |
|--------|------|
| **sentence-transformers** (`all-mpnet-base-v2`) | Embeds full turns and individual sentences into the working space used for compression and hallucination checks |
| **Ground-truth anchors** (prompt + answer pairs from raw history) | Pulls vectors toward verified conversational facts so the space stays aligned with what actually happened |
| **Topological deformation logic** (implemented in our scoring/elimination flow) | Uses sentence-level structure to reshape local neighborhoods and separate stable context from likely hallucinated drift |
| **Hallucination-aware embedding baseline** (pretrained separation objective) | Starts from a space tuned to discriminate hallucinated vs grounded content before online updates |
| **transformers** (`facebook/bart-large-cnn`, `AutoTokenizer`) | Local paragraph generation + token counting for MMR budget |
| **NumPy** | Cosine similarity and MMR from scratch; also supports geometry updates without external MMR tooling |
| **python-dotenv** (optional) | Loads `.env` for `HF_TOKEN` / hub token aliases |

Frontend (e.g. React) can expose a **pipeline panel** so each stage is visible in real time during demos: compression -> elimination -> reconstruction, plus the evolving embedding map.

---

## Why “MemGuard”

The name implies **purification** — pulling something **essential** out of noise. The pipeline is exactly that: **compress** for signal, **eliminate** what the signal cannot support, **reconstruct** what remains into context the model can trust.
