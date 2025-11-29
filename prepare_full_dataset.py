"""
Prepare Full Custom Dataset from vocab_src

Combines:
- Text files (books, history, etc.)
- JSONL files (conversations, grammar, emotions, intents, instructions)

Creates a comprehensive training dataset for Aura.
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import torch

def load_jsonl_texts(filepath: str, text_keys: List[str] = None) -> List[str]:
    """
    Load texts from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        text_keys: Keys to extract text from (tries common keys if None)
    """
    texts = []
    
    if text_keys is None:
        text_keys = ['text', 'content', 'input', 'output', 'question', 'answer', 
                     'correct_example', 'incorrect_example', 'explanation', 'turns']
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract text from various keys
                    for key in text_keys:
                        if key in data:
                            value = data[key]
                            if isinstance(value, str) and len(value) > 10:
                                texts.append(value)
                            elif isinstance(value, list):
                                # Handle conversation turns
                                for item in value:
                                    if isinstance(item, str) and len(item) > 10:
                                        texts.append(item)
                                    elif isinstance(item, dict):
                                        for v in item.values():
                                            if isinstance(v, str) and len(v) > 10:
                                                texts.append(v)
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        
    return texts


def load_txt_files(vocab_dir: str, min_length: int = 100) -> List[str]:
    """Load all text files from directory."""
    texts = []
    vocab_path = Path(vocab_dir)
    txt_files = list(vocab_path.glob("*.txt"))
    
    print(f"Found {len(txt_files)} text files")
    
    for filepath in tqdm(txt_files, desc="Loading text files"):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                
            if len(content) < min_length:
                continue
                
            # Split into paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) >= min_length:
                    texts.append(para)
                    
        except Exception as e:
            continue
    
    return texts


def load_all_sources(vocab_dir: str = "vocab_src") -> Dict[str, List[str]]:
    """
    Load all data sources.
    
    Returns dict with source name -> list of texts
    """
    sources = {}
    vocab_path = Path(vocab_dir)
    
    # JSONL files
    jsonl_files = {
        'conversation': 'conversation.jsonl',
        'grammar': 'combined_grammar.jsonl',
        'emotions': 'emotions.jsonl',
        'emotion_valence': 'emotion_valence_arousal_realm_phase.jsonl',
        'historical_facts': 'historical_facts.jsonl',
        'intents': 'intent_all.jsonl',
        'instructions': 'instruct_55k_clean.jsonl',
        'greetings': 'greetings.jsonl',
        'events': 'conversations_individual_events.jsonl',
    }
    
    for name, filename in jsonl_files.items():
        filepath = vocab_path / filename
        if filepath.exists():
            print(f"\nLoading {name} from {filename}...")
            texts = load_jsonl_texts(str(filepath))
            if texts:
                sources[name] = texts
                print(f"  Loaded {len(texts)} texts")
    
    # Text files
    print(f"\nLoading text files...")
    txt_texts = load_txt_files(vocab_dir)
    if txt_texts:
        sources['books'] = txt_texts
        print(f"  Loaded {len(txt_texts)} text segments")
    
    return sources


def tokenize_all(sources: Dict[str, List[str]], tokenizer) -> List[int]:
    """Tokenize all sources into a single token list."""
    all_tokens = []
    
    for name, texts in sources.items():
        print(f"\nTokenizing {name} ({len(texts)} texts)...")
        
        for text in tqdm(texts, desc=f"  {name}"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) >= 5:
                all_tokens.extend(tokens)
                # Add separator between texts
                all_tokens.append(tokenizer.eos_token_id or 1)
    
    return all_tokens


def create_sequences(tokens: List[int], seq_length: int = 256) -> torch.Tensor:
    """Create fixed-length sequences."""
    num_sequences = len(tokens) // seq_length
    tokens = tokens[:num_sequences * seq_length]
    sequences = torch.tensor(tokens).reshape(num_sequences, seq_length)
    
    # Shuffle
    perm = torch.randperm(len(sequences))
    sequences = sequences[perm]
    
    return sequences


def prepare_full_dataset(
    vocab_dir: str = "vocab_src",
    output_path: str = "aura_full_dataset.pt",
    seq_length: int = 256,
    tokenizer_name: str = "google/flan-t5-base"
):
    """
    Full pipeline to prepare comprehensive dataset.
    """
    print("=" * 60)
    print("PREPARING FULL AURA DATASET")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, legacy=True)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Load all sources
    print("\n" + "-" * 40)
    print("LOADING DATA SOURCES")
    print("-" * 40)
    sources = load_all_sources(vocab_dir)
    
    # Summary
    print("\n" + "-" * 40)
    print("SOURCE SUMMARY")
    print("-" * 40)
    total_texts = 0
    for name, texts in sources.items():
        print(f"  {name}: {len(texts):,} texts")
        total_texts += len(texts)
    print(f"  TOTAL: {total_texts:,} texts")
    
    # Tokenize
    print("\n" + "-" * 40)
    print("TOKENIZING")
    print("-" * 40)
    all_tokens = tokenize_all(sources, tokenizer)
    print(f"\nTotal tokens: {len(all_tokens):,}")
    
    # Create sequences
    print("\n" + "-" * 40)
    print("CREATING SEQUENCES")
    print("-" * 40)
    sequences = create_sequences(all_tokens, seq_length)
    print(f"Created {len(sequences):,} sequences of length {seq_length}")
    
    # Save
    print("\n" + "-" * 40)
    print("SAVING")
    print("-" * 40)
    torch.save({
        'sequences': sequences,
        'vocab_size': tokenizer.vocab_size,
        'seq_length': seq_length,
        'sources': {k: len(v) for k, v in sources.items()},
        'total_tokens': len(all_tokens)
    }, output_path)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")
    
    # Final stats
    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)
    print(f"Sequences: {len(sequences):,}")
    print(f"Tokens: {len(all_tokens):,}")
    print(f"Seq length: {seq_length}")
    
    steps_32 = len(sequences) // 32
    print(f"\nTraining estimates (batch=32):")
    print(f"  Steps per epoch: {steps_32:,}")
    print(f"  5K steps = {5000/steps_32:.2f} epochs")
    print(f"  20K steps = {20000/steps_32:.2f} epochs")
    print(f"  50K steps = {50000/steps_32:.2f} epochs")
    
    return sequences


# Colab loader code
COLAB_LOADER = '''
# ============================================================
# CUSTOM DATASET LOADER FOR COLAB
# ============================================================

def load_aura_dataset(dataset_path, config, device):
    """
    Load pre-tokenized Aura dataset.
    
    Upload aura_full_dataset.pt to Google Drive first.
    """
    print(f"Loading dataset from {dataset_path}...")
    data = torch.load(dataset_path)
    
    sequences = data['sequences']
    print(f"Loaded {len(sequences):,} sequences")
    print(f"Sources: {data['sources']}")
    print(f"Total tokens: {data['total_tokens']:,}")
    
    # Create batches
    batches = []
    batch_size = config.batch_size
    
    # Shuffle
    perm = torch.randperm(len(sequences))
    sequences = sequences[perm]
    
    for i in range(0, len(sequences) - batch_size, batch_size):
        batch = sequences[i:i+batch_size].to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        batches.append((input_ids, labels))
    
    print(f"Created {len(batches)} batches")
    return batches

# Usage in main():
# batches = load_aura_dataset('/content/drive/MyDrive/aura_full_dataset.pt', config, device)
'''


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare full Aura dataset')
    parser.add_argument('--vocab_dir', default='vocab_src', help='Directory with source files')
    parser.add_argument('--output', default='aura_full_dataset.pt', help='Output file')
    parser.add_argument('--seq_length', type=int, default=256, help='Sequence length')
    
    args = parser.parse_args()
    
    sequences = prepare_full_dataset(
        vocab_dir=args.vocab_dir,
        output_path=args.output,
        seq_length=args.seq_length
    )
    
    print("\n" + "=" * 60)
    print("COLAB LOADER CODE")
    print("=" * 60)
    print(COLAB_LOADER)

