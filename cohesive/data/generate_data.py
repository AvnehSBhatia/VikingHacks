"""
generate_data.py
────────────────
Generates synthetic hallucination training data from the DAHacks schema:

    dialogue_history | right_response | hallucinated_response

Two modes:
    1. --from-csv:   augment an existing CSV by generating more hallucinated
                     variants of existing right responses using Claude API
    2. --synthetic:  generate entirely synthetic dialogue/response pairs

Output: CSV with the 3-column schema shown in the dataset image.

Usage
─────
    python generate_data.py --from-csv data/raw.csv --out data/train.csv --n 1000
    python generate_data.py --synthetic --out data/synthetic.csv --n 500
"""

from __future__ import annotations

import os
import csv
import json
import random
import argparse
import re
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Hallucination patterns
# ══════════════════════════════════════════════════════════════════════════════

HALLUCINATION_STRATEGIES = [
    "swap_entity",       # replace a named entity with a plausible wrong one
    "wrong_attribute",   # change a factual attribute (date, number, place)
    "false_negation",    # negate a true claim
    "invented_fact",     # add a plausible but false fact
    "wrong_relation",    # swap who did what to whom
]

# Some factual swaps for entity-level hallucination
ENTITY_SWAPS = {
    "Robert Downey Jr.": ["Tom Hanks", "Brad Pitt", "Matt Damon", "Leonardo DiCaprio"],
    "Jake Gyllenhaal":   ["Ryan Reynolds", "Chris Evans", "Michael B. Jordan"],
    "Tom Hanks":         ["Robert Downey Jr.", "Denzel Washington", "Morgan Freeman"],
    "Zodiac":            ["Se7en", "The Town", "Prisoners", "Mindhunter"],
    "crime fiction":     ["romantic comedy", "science fiction", "horror", "documentary"],
}

SYNTHETIC_DIALOGUES = [
    {
        "dialogue": "[Human]: Do you know who directed Inception?\n[Assistant]: Christopher Nolan directed it.\n[Human]: Right, and he also did The Dark Knight?",
        "right": "Yes, Christopher Nolan directed The Dark Knight Rises and the whole Batman trilogy.",
        "hallu_templates": [
            "Yes, Christopher Nolan directed The Dark Knight, along with Interstellar and Memento.",
            "Actually, it was Zack Snyder who directed The Dark Knight — Nolan only produced it.",
        ]
    },
    {
        "dialogue": "[Human]: What year did World War II end?\n[Assistant]: 1945.\n[Human]: And who signed the Japanese surrender?",
        "right": "General Douglas MacArthur presided over the formal Japanese surrender ceremony on September 2, 1945.",
        "hallu_templates": [
            "General Dwight Eisenhower signed the Japanese surrender on August 15, 1945.",
            "President Harry Truman personally signed the surrender documents aboard the USS Missouri.",
        ]
    },
    {
        "dialogue": "[Human]: I love Taylor Swift\n[Assistant]: Great taste!\n[Human]: What's her most recent album?",
        "right": "As of my knowledge cutoff, The Tortured Poets Department was her most recent album, released in April 2024.",
        "hallu_templates": [
            "Her most recent album is Midnights, which she released in 2023.",
            "She just released Eras, her newest album, which came out in early 2024.",
        ]
    },
    {
        "dialogue": "[Human]: Can you explain photosynthesis?\n[Assistant]: It converts sunlight to energy.\n[Human]: What's the main molecule involved?",
        "right": "Chlorophyll is the primary pigment that absorbs light; glucose is the main energy molecule produced.",
        "hallu_templates": [
            "ATP synthase is the main molecule — it directly converts sunlight into chemical energy.",
            "Hemoglobin is the key molecule in photosynthesis, carrying carbon dioxide to the chloroplasts.",
        ]
    },
    {
        "dialogue": "[Human]: I'm reading Crime and Punishment\n[Assistant]: Great book!\n[Human]: Who wrote it?",
        "right": "Fyodor Dostoevsky wrote Crime and Punishment, published in 1866.",
        "hallu_templates": [
            "Leo Tolstoy wrote Crime and Punishment, one of his most celebrated works alongside War and Peace.",
            "Crime and Punishment was written by Ivan Turgenev and published in 1867.",
        ]
    },
    {
        "dialogue": "[Human]: I'm learning Python\n[Assistant]: Awesome!\n[Human]: What's a list comprehension?",
        "right": "A list comprehension is a concise way to create lists: [x*2 for x in range(10)] gives [0,2,4,...,18].",
        "hallu_templates": [
            "A list comprehension is a special dictionary type that maps keys to lists of values.",
            "List comprehensions are a C++ feature ported to Python 3.9 for faster iteration.",
        ]
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Simple rule-based hallucination generator (no API needed)
# ══════════════════════════════════════════════════════════════════════════════

def hallucinate_text(text: str, strategy: str | None = None) -> str:
    if strategy is None:
        strategy = random.choice(HALLUCINATION_STRATEGIES)

    if strategy == "swap_entity":
        for entity, swaps in ENTITY_SWAPS.items():
            if entity in text:
                replacement = random.choice(swaps)
                return text.replace(entity, replacement, 1)
        # fallback: wrong_attribute
        strategy = "wrong_attribute"

    if strategy == "wrong_attribute":
        # replace years
        text = re.sub(
            r'\b(19|20)\d{2}\b',
            lambda m: str(int(m.group(0)) + random.choice([-3, -2, 2, 3, 5])),
            text
        )
        # replace numbers
        text = re.sub(
            r'\b([1-9]\d{1,3})\b',
            lambda m: str(int(m.group(0)) + random.choice([-10, 10, 100, -100])),
            text
        )
        return text

    if strategy == "false_negation":
        # insert "not" before a verb
        text = re.sub(r'\b(was|is|are|were|did|does|do|has|have)\b',
                      lambda m: m.group(0) + " not", text, count=1)
        return text

    if strategy == "invented_fact":
        additions = [
            " Additionally, this was later disputed by scholars.",
            " Interestingly, this occurred exactly 10 years after the French Revolution.",
            " Tom Hanks was also involved in this project.",
            " This was later adapted into a Netflix series.",
        ]
        return text + random.choice(additions)

    if strategy == "wrong_relation":
        # crude: swap first two proper nouns
        words = text.split()
        proper = [i for i, w in enumerate(words) if w[0].isupper() and len(w) > 2]
        if len(proper) >= 2:
            words[proper[0]], words[proper[1]] = words[proper[1]], words[proper[0]]
            return " ".join(words)

    return text + " (This claim is incorrect.)"


# ══════════════════════════════════════════════════════════════════════════════
# Generators
# ══════════════════════════════════════════════════════════════════════════════

def generate_from_csv(
    input_csv: str,
    output_csv: str,
    n: int,
) -> None:
    import pandas as pd
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in df.iterrows():
        dlg   = str(row.get("dialogue_history", ""))
        right = str(row.get("right_response", ""))
        hallu = str(row.get("hallucinated_response", ""))

        # keep original
        rows.append({"dialogue_history": dlg, "right_response": right, "hallucinated_response": hallu})

        # augment with generated hallucinations
        for _ in range(2):
            h = hallucinate_text(right)
            rows.append({"dialogue_history": dlg, "right_response": right, "hallucinated_response": h})

        if len(rows) >= n:
            break

    random.shuffle(rows)
    rows = rows[:n]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[DataGen] Wrote {len(rows)} rows to {output_csv}")


def generate_synthetic(output_csv: str, n: int) -> None:
    import pandas as pd

    rows = []
    while len(rows) < n:
        template = random.choice(SYNTHETIC_DIALOGUES)
        dlg   = template["dialogue"]
        right = template["right"]

        # use pre-written hallu or generate one
        if template["hallu_templates"] and random.random() < 0.5:
            hallu = random.choice(template["hallu_templates"])
        else:
            hallu = hallucinate_text(right)

        rows.append({
            "dialogue_history":      dlg,
            "right_response":        right,
            "hallucinated_response": hallu,
        })

    random.shuffle(rows)
    pd.DataFrame(rows[:n]).to_csv(output_csv, index=False)
    print(f"[DataGen] Wrote {n} synthetic rows to {output_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-csv",   default=None, help="Augment existing CSV")
    p.add_argument("--synthetic",  action="store_true", help="Generate synthetic data")
    p.add_argument("--out",        required=True)
    p.add_argument("--n",          type=int, default=1000)
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        generate_from_csv(args.from_csv, args.out, args.n)
    elif args.synthetic:
        generate_synthetic(args.out, args.n)
    else:
        print("Specify --from-csv or --synthetic")