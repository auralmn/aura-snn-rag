"""
Prepare Custom Dataset from vocab_src

Creates a tokenized dataset from your local text files for training Aura.
This gives you domain-specific training data (history, science, etc.)
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import torch

def load_all_texts(vocab_dir: str = "vocab_src", min_length: int = 100) -> List[str]:
    """
    Load all text files from vocab_src directory.
    
    Args:
        vocab_dir: Directory containing text files
        min_length: Minimum character length to include
        
    Returns:
        List of text strings
    """
    texts = []
    vocab_path = Path(vocab_dir)
    
    # Get all .txt files
    txt_files = list(vocab_path.glob("*.txt"))
    print(f"Found {len(txt_files)} text files")
    
    for filepath in tqdm(txt_files, desc="Loading texts"):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Skip very short files
            if len(content) < min_length:
                continue
                
            # Clean up the text
            content = content.strip()
            
            # Split into paragraphs (better for training)
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) >= min_length:
                    texts.append(para)
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    print(f"Loaded {len(texts)} text segments")
    return texts


def tokenize_texts(texts: List[str], tokenizer, max_length: int = 512) -> List[List[int]]:
    """
    Tokenize all texts.
    
    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of token ID lists
    """
    all_tokens = []
    
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= 10:  # Skip very short sequences
            all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    return all_tokens


def create_sequences(tokens: List[int], seq_length: int = 256) -> torch.Tensor:
    """
    Create fixed-length sequences from token list.
    
    Args:
        tokens: Flat list of token IDs
        seq_length: Sequence length
        
    Returns:
        Tensor of shape [num_sequences, seq_length]
    """
    num_sequences = len(tokens) // seq_length
    tokens = tokens[:num_sequences * seq_length]
    sequences = torch.tensor(tokens).reshape(num_sequences, seq_length)
    
    # Shuffle
    perm = torch.randperm(len(sequences))
    sequences = sequences[perm]
    
    print(f"Created {len(sequences):,} sequences of length {seq_length}")
    return sequences


def save_dataset(sequences: torch.Tensor, output_path: str = "vocab_dataset.pt"):
    """Save tokenized dataset to file."""
    torch.save(sequences, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def prepare_dataset(
    vocab_dir: str = "vocab_src",
    output_path: str = "vocab_dataset.pt",
    seq_length: int = 256,
    tokenizer_name: str = "google/flan-t5-base"
):
    """
    Full pipeline to prepare dataset.
    
    Args:
        vocab_dir: Directory with text files
        output_path: Where to save the dataset
        seq_length: Sequence length for training
        tokenizer_name: HuggingFace tokenizer to use
    """
    print("=" * 60)
    print("PREPARING CUSTOM DATASET")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Load texts
    print(f"\nLoading texts from {vocab_dir}...")
    texts = load_all_texts(vocab_dir)
    
    # Tokenize
    print("\nTokenizing...")
    tokens = tokenize_texts(texts, tokenizer)
    
    # Create sequences
    print(f"\nCreating sequences (length={seq_length})...")
    sequences = create_sequences(tokens, seq_length)
    
    # Save
    print("\nSaving...")
    save_dataset(sequences, output_path)
    
    # Stats
    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    print(f"Source files: {len(list(Path(vocab_dir).glob('*.txt')))}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Sequences: {len(sequences):,}")
    print(f"Sequence length: {seq_length}")
    print(f"Effective training tokens: {len(sequences) * seq_length:,}")
    
    # Estimate training time
    steps_per_epoch = len(sequences) // 32  # Assuming batch_size=32
    print(f"\nEstimated steps per epoch (batch=32): {steps_per_epoch:,}")
    print(f"For 5K steps: {5000 / steps_per_epoch:.1f} epochs")
    print(f"For 20K steps: {20000 / steps_per_epoch:.1f} epochs")
    print(f"For 50K steps: {50000 / steps_per_epoch:.1f} epochs")
    
    return sequences


def create_colab_loader_code():
    """Generate code to load this dataset in Colab."""
    code = '''
# Add this to your Colab notebook to use the custom dataset:

def load_custom_dataset(dataset_path, config, device):
    """Load pre-tokenized custom dataset."""
    print(f"Loading custom dataset from {dataset_path}...")
    sequences = torch.load(dataset_path)
    print(f"Loaded {len(sequences):,} sequences")
    
    # Create batches
    batches = []
    batch_size = config.batch_size
    
    for i in range(0, len(sequences) - batch_size, batch_size):
        batch = sequences[i:i+batch_size].to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        batches.append((input_ids, labels))
    
    print(f"Created {len(batches)} batches")
    return batches

# In main():
# batches = load_custom_dataset('/content/drive/MyDrive/vocab_dataset.pt', config, device)
'''
    print(code)
    return code


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare custom dataset from vocab_src')
    parser.add_argument('--vocab_dir', default='vocab_src', help='Directory with text files')
    parser.add_argument('--output', default='vocab_dataset.pt', help='Output file path')
    parser.add_argument('--seq_length', type=int, default=256, help='Sequence length')
    parser.add_argument('--tokenizer', default='google/flan-t5-base', help='Tokenizer name')
    
    args = parser.parse_args()
    
    sequences = prepare_dataset(
        vocab_dir=args.vocab_dir,
        output_path=args.output,
        seq_length=args.seq_length,
        tokenizer_name=args.tokenizer
    )
    
    print("\n" + "=" * 60)
    print("COLAB LOADER CODE")
    print("=" * 60)
    create_colab_loader_code()

