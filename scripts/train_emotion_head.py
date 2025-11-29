"""
Train the Emotion/Intent/Tone/Personality head on vocab_src JSONL corpora using FLAN-T5 embeddings.

This is a lightweight trainer that:
- Loads texts/labels from:
    * vocab_src/amygdala_affect.jsonl
    * vocab_src/emotion_valence_arousal_realm_phase.jsonl
    * vocab_src/identitya.jsonl
- Tokenizes with FLAN-T5 tokenizer, encodes with FLAN-T5 encoder (no generation).
- Pools hidden states, then trains the EmotionPersonalityHead.
- Saves weights and label maps to repo/models/emotion_head/.

Usage (from repo root):
    python scripts/train_emotion_head.py --epochs 3 --max-examples 5000

Note: This is a reference script; adjust batch sizes/epochs for your GPU.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from prosody.emotion_head import EmotionPersonalityHead, EmotionPersonalityLoss, pool_token_embeddings


class EmotionDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec.get("text") or rec.get("input") or rec.get("prompt") or ""
        return text, rec

    def collate_fn(self, batch):
        texts = [b[0] for b in batch]
        recs = [b[1] for b in batch]
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return tok, recs


def load_jsonl(paths: List[Path], max_examples: int = None) -> List[Dict]:
    records = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                    if max_examples and len(records) >= max_examples:
                        return records
                except Exception:
                    continue
    return records


def build_label_maps(records: List[Dict]) -> Dict[str, Dict]:
    emotions, intents, tones, personas = set(), set(), set(), set()
    for r in records:
        emotions.add(r.get("emotion", r.get("primary", "none")) or "none")
        intents.add(r.get("intent", "none"))
        tones.add(r.get("tone", r.get("realm", "none")))
        personas.add(r.get("identity", r.get("persona", "none")))
    def mk_map(vals):
        vals = sorted(list(vals))
        if "none" not in vals:
            vals = ["none"] + vals
        return {v: i for i, v in enumerate(vals)}, {i: v for i, v in enumerate(vals)}
    emo_fwd, emo_rev = mk_map(emotions)
    intent_fwd, intent_rev = mk_map(intents)
    tone_fwd, tone_rev = mk_map(tones)
    persona_fwd, persona_rev = mk_map(personas)
    return {
        "emotion": emo_fwd,
        "intent": intent_fwd,
        "tone": tone_fwd,
        "personality": persona_fwd,
        "emotion_rev": emo_rev,
        "intent_rev": intent_rev,
        "tone_rev": tone_rev,
        "personality_rev": persona_rev,
    }


def to_targets(batch_recs: List[Dict], label_maps: Dict[str, Dict], device) -> Dict[str, torch.Tensor]:
    emo = []
    intent = []
    tone = []
    persona = []
    for r in batch_recs:
        emo.append(label_maps["emotion"].get(r.get("emotion", r.get("primary", "none")), 0))
        intent.append(label_maps["intent"].get(r.get("intent", "none"), 0))
        tone.append(label_maps["tone"].get(r.get("tone", r.get("realm", "none")), 0))
        persona.append(label_maps["personality"].get(r.get("identity", r.get("persona", "none")), 0))
    return {
        "emotion": torch.tensor(emo, device=device, dtype=torch.long),
        "intent": torch.tensor(intent, device=device, dtype=torch.long),
        "tone": torch.tensor(tone, device=device, dtype=torch.long),
        "personality": torch.tensor(persona, device=device, dtype=torch.long),
    }


def train(args):
    repo_root = Path(__file__).resolve().parents[1]
    data_paths = [
        repo_root / "vocab_src" / "amygdala_affect.jsonl",
        repo_root / "vocab_src" / "emotion_valence_arousal_realm_phase.jsonl",
        repo_root / "vocab_src" / "identitya.jsonl",
    ]
    records = load_jsonl(data_paths, max_examples=args.max_examples)
    if not records:
        print("No data found in vocab_src/*.jsonl for emotion training.")
        return

    label_maps = build_label_maps(records)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=True)
    encoder = AutoModel.from_pretrained("google/flan-t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    dataset = EmotionDataset(records, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    head = EmotionPersonalityHead(
        embed_dim=encoder.config.d_model,
        num_emotions=len(label_maps["emotion"]),
        num_intents=len(label_maps["intent"]),
        num_tones=len(label_maps["tone"]),
        num_personalities=len(label_maps["personality"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = EmotionPersonalityLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        total_batches = 0
        for tok_batch, recs in loader:
            tok_batch = {k: v.to(device) for k, v in tok_batch.items()}
            with torch.no_grad():
                enc_out = encoder(**tok_batch).last_hidden_state
            pooled = pool_token_embeddings(enc_out, tok_batch.get("attention_mask"))
            logits = head(pooled)
            targets = to_targets(recs, label_maps, device)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(1, total_batches)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Save
    out_dir = repo_root / "models" / "emotion_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_maps": label_maps,
            "config": {
                "embed_dim": encoder.config.d_model,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
        },
        out_dir / "emotion_head.pt",
    )
    print(f"Saved emotion/personality head to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-examples", type=int, default=10000)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
