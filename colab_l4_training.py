# -*- coding: utf-8 -*-
"""
Aura SNN-RAG Training Script for Google Colab L4 (20GB VRAM)

This script trains the optimized Aura HippocampalTransformer with:
- SNN-based FFN layers (biological plausibility)
- RAG memory retrieval (context-aware generation)
- Gradient checkpointing (memory efficiency)
- Cosine LR with warmup
- WikiText-2 dataset (fast iteration for testing)

Copy and paste this entire script into a Colab notebook cell.
"""

# ============================================================
# CELL 1: Setup & Dependencies
# ============================================================
# !pip install -q datasets transformers sentencepiece torch tqdm

# Clone repository
# !git clone https://github.com/auralmn/aura-hybrid-pre-model.git
# !cd aura-hybrid-pre-model && git checkout master && git pull

import sys
import os
import gc
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Ensure repo core modules are importable (works in Colab or local)
REPO_SRC = Path(__file__).resolve().parent / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))

from core.hippocampal import HippocampalFormation
from core.language_zone.hippocampal_transformer import HippocampalTransformer
from core.limbic_system import Amygdala
from core.endocrine import EndocrineSystem
from core.thalamus import Thalamus
# Optional continuous learning (lazy imports inside helper)
from typing import Any

# ============================================================
# CELL 2: Configuration
# ============================================================

@dataclass
class Config:
    """Optimized config for L4 GPU (20GB VRAM)."""
    
    # === MODEL ===
    vocab_size: int = 32000
    embedding_dim: int = 512      # Reduced for faster testing
    num_layers: int = 6           # Reduced for faster testing
    num_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.1
    max_seq_len: int = 256        # Reduced for faster testing
    intermediate_size: int = 2048
    
    # === HIPPOCAMPAL ===
    theta_frequency: float = 8.0
    gamma_frequency: float = 40.0
    n_place_cells: int = 500      # Reduced for testing
    n_time_cells: int = 50
    n_grid_cells: int = 50
    place_cell_sparsity: float = 0.03
    
    # === SNN ===
    use_snn_ffn: bool = True
    snn_layers: List[int] = None  # Will be set to [0, 2, 4] in __post_init__
    snn_timesteps: int = 4
    snn_L: int = 8
    
    # === RAG ===
    use_rag: bool = True
    memory_injection: str = "gate"  # "gate", "cross_attention", "concat"
    num_retrieved: int = 3
    max_memories: int = 10000
    
    # === TRAINING ===
    batch_size: int = 8
    gradient_accumulation: int = 4
    lr: float = 3e-4              # Increased for faster learning
    warmup_steps: int = 200       # Shorter warmup
    max_steps: int = 5000         # Short run for testing
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # === MEMORY WARMUP ===
    memory_warmup_steps: int = 500  # Reduced - let model learn first
    min_lr_ratio: float = 0.1
    
    # === CONSOLIDATION ===
    sleep_interval: int = 1000
    ewc_lambda: float = 0.4
    replay_buffer_size: int = 10000
    
    # === LOSS ===
    label_smoothing: float = 0.1
    entropy_lambda: float = 0.05
    sparsity_lambda: float = 0.02
    target_sparsity: float = 0.03
    
    # === OPTIMIZATION ===
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False  # disable to avoid checkpoint tensor-mismatch on Colab
    eval_interval: int = 100
    save_interval: int = 500
    enable_continuous_learning: bool = False  # set True to build orchestrator (manual start)
    enable_amygdala: bool = True
    enable_endocrine: bool = True
    enable_thalamus: bool = True
    enable_centroid_index: bool = True
    checkpoint_path: str = "models/checkpoint_final.pt"
    load_optimizer: bool = False
    
    def __post_init__(self):
        if self.snn_layers is None:
            self.snn_layers = list(range(0, self.num_layers, 2))


# Preset configurations
def get_test_config():
    """Quick test config (5K steps, ~15 min)."""
    return Config()

def get_baseline_config():
    """Baseline config WITHOUT SNN - for debugging learning issues."""
    config = Config()
    config.use_snn_ffn = False  # Disable SNN
    config.use_rag = False      # Disable RAG
    config.snn_layers = []      # No SNN layers
    config.lr = 5e-4            # Higher LR for faster convergence
    config.warmup_steps = 100
    config.memory_warmup_steps = 0
    return config

def get_medium_config():
    """Medium config for longer training (20K steps, ~2 hours)."""
    config = Config()
    config.embedding_dim = 768
    config.num_layers = 8
    config.num_heads = 12
    config.intermediate_size = 3072
    config.max_seq_len = 384
    config.max_steps = 20000
    config.warmup_steps = 1000
    config.memory_warmup_steps = 2000
    config.n_place_cells = 1000
    config.max_memories = 50000
    config.snn_layers = [0, 2, 4, 6]
    return config

def get_full_config():
    """Full config for production training (50K+ steps, ~6 hours)."""
    config = Config()
    config.embedding_dim = 768
    config.num_layers = 12
    config.num_heads = 12
    config.intermediate_size = 3072
    config.max_seq_len = 512
    config.max_steps = 50000
    config.warmup_steps = 2000
    config.memory_warmup_steps = 5000
    config.n_place_cells = 2000
    config.max_memories = 100000
    config.batch_size = 12
    config.gradient_accumulation = 4
    config.snn_layers = [0, 2, 4, 6, 8, 10]
    return config


# ============================================================
# CELL 3: Helpers for interactive hippocampal memory
# ============================================================

def store_custom_memory(hippocampus, features: torch.Tensor, memory_id: Optional[str] = None):
    """Store an external feature vector into the hippocampal bank."""
    if hippocampus is None:
        return
    feat = features.detach()
    if feat.dim() == 2:
        feat = feat.mean(dim=0)
    if memory_id is None:
        memory_id = f"external-{int(time.time())}"
    hippocampus.create_episodic_memory(memory_id=memory_id, event_id=memory_id, features=feat)
    return memory_id


def retrieve_custom_memories(hippocampus, query_features: torch.Tensor, location: Optional[torch.Tensor] = None, k: int = 5):
    """Retrieve top-k similar memories for debugging or interactive use."""
    if hippocampus is None:
        return []
    if query_features.dim() > 1:
        query_features = query_features.mean(dim=0)
    return hippocampus.retrieve_similar_memories(query_features, location=location, k=k)


def one_shot_memorize_text(text: str, tokenizer, model, hippocampus, device, memory_id: Optional[str] = None):
    """
    One-shot learning helper: encode a support text and store its embedding into episodic memory.
    Retrieval happens automatically on future forwards with use_memory=True.
    """
    if hippocampus is None or model is None or tokenizer is None:
        return None
    ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=getattr(model.config, "max_seq_len", 256)).to(device)
    mem_id = memory_id or f"oneshot-{int(time.time())}"
    model.eval()
    with torch.no_grad():
        # Store at the end of forward pass using store_memory flag
        _logits, _ = model(ids, prosody=None, use_memory=False, store_memory=True, memory_ids=[mem_id])
    return mem_id


def one_shot_memorize_and_generate(
    support_text: str,
    prompt: str,
    tokenizer,
    model,
    hippocampus,
    device,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
) -> str:
    """
    Convenience helper for Colab testing:
      1) Stores support_text into hippocampal episodic memory.
      2) Generates continuation for prompt with memory retrieval on.
    """
    mem_id = one_shot_memorize_text(support_text, tokenizer, model, hippocampus, device)
    # Simple sampling loop (matches sample_generate in main)
    model.eval()
    generated = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -getattr(model.config, "max_seq_len", 256):]
            logits, _ = model(context, use_memory=True, store_memory=False)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def build_prosody(amygdala: Amygdala, model: HippocampalTransformer, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute prosody tensor [B, L, 4] from Amygdala arousal/valence on token embeddings.
    Uses no_grad to avoid coupling gradients.
    """
    with torch.no_grad():
        token_embeds = model.semantic_encoder.token_embedding(input_ids)
        limbic = amygdala(token_embeds)
        arousal = torch.tensor(limbic["arousal"], device=input_ids.device, dtype=token_embeds.dtype)
        valence = torch.tensor(limbic["valence"], device=input_ids.device, dtype=token_embeds.dtype)
        pros = torch.stack([arousal, valence, arousal, valence])
        prosody = pros.view(1, 1, 4).expand(input_ids.size(0), input_ids.size(1), 4)
    return prosody


def ingest_jsonl_to_memory(
    path: str,
    tokenizer,
    model,
    hippocampus,
    device,
    max_items: int = 1000,
) -> int:
    """
    Stream a JSONL file and store examples into episodic memory.
    Flexible fields per line: 'text', or pairs like (instruction/output), (prompt/completion), (input/output).
    """
    import json
    if hippocampus is None or model is None or tokenizer is None:
        return 0
    stored = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if stored >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = None
            if isinstance(obj, str):
                text = obj
            elif isinstance(obj, dict):
                if "text" in obj:
                    text = obj["text"]
                elif "instruction" in obj and "output" in obj:
                    text = f"Instruction: {obj['instruction']}\nResponse: {obj.get('output','')}"
                elif "prompt" in obj and "completion" in obj:
                    text = f"Prompt: {obj['prompt']}\nCompletion: {obj.get('completion','')}"
                elif "input" in obj and "output" in obj:
                    text = f"Input: {obj['input']}\nOutput: {obj.get('output','')}"
            if not text:
                continue
            mem_id = f"jsonl-{stored}"
            one_shot_memorize_text(text, tokenizer, model, hippocampus, device, memory_id=mem_id)
            stored += 1
    return stored


def ingest_csv_pairs_to_memory(
    path: str,
    tokenizer,
    model,
    hippocampus,
    device,
    max_items: int = 1000,
    delimiter: str = ",",
) -> int:
    """
    Stream a CSV with two columns (e.g., question, answer) and store as episodic memories.
    Useful for timeline_conversations.csv or other Q/A pairs.
    """
    import csv
    if hippocampus is None or model is None or tokenizer is None:
        return 0
    stored = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if stored >= max_items:
                break
            if len(row) < 2:
                continue
            q, a = row[0].strip(), row[1].strip()
            if not q and not a:
                continue
            text = f"Question: {q}\nAnswer: {a}"
            mem_id = f"csv-{stored}"
            one_shot_memorize_text(text, tokenizer, model, hippocampus, device, memory_id=mem_id)
            stored += 1
    return stored


def build_continuous_learning_orchestrator(config, hippocampus, tokenizer=None, model=None, device=None) -> Optional[Any]:
    """
    Lazily build a ContinuousLearningOrchestrator in memory-only mode.
    Caller is responsible for awaiting orchestrator.start() in an event loop.
    """
    if not getattr(config, "enable_continuous_learning", False):
        return None
    try:
        from base.events import EventBus
        from services.continuous_learning import ContinuousLearningOrchestrator, create_default_feeds
    except Exception as e:
        print(f"[CL] Continuous learning not available: {e}")
        return None
    try:
        event_bus = EventBus()
        embed_fn = None
        if tokenizer is not None and model is not None and hasattr(model, "semantic_encoder"):
            token_embed = model.semantic_encoder.token_embedding
            max_len = min(getattr(config, "max_seq_len", 256), 256)
            def _embed(text: str):
                ids = tokenizer.encode(text, truncation=True, max_length=max_len)
                if not ids:
                    ids = [tokenizer.pad_token_id or 0]
                t = torch.tensor(ids, device=device or token_embed.weight.device).unsqueeze(0)
                with torch.no_grad():
                    return token_embed(t).mean(dim=1).squeeze(0)
            embed_fn = _embed

        # processor=None in memory_only mode; orchestrator will just push memories
        orchestrator = ContinuousLearningOrchestrator(
            processor=None,
            event_bus=event_bus,
            hippocampus=hippocampus,
            memory_only=True,
            tokenizer=tokenizer,
            embed_fn=embed_fn
        )
        for fd in create_default_feeds():
            orchestrator.add_feed(fd)
        print("[CL] Continuous learning orchestrator built (memory-only). Call `await orchestrator.start()` in a separate cell to run.")
        return orchestrator
    except Exception as e:
        print(f"[CL] Failed to build orchestrator: {e}")
        return None


# ============================================================
# CELL 4: Loss Function
# ============================================================

class AuraLoss(nn.Module):
    """Loss with entropy regularization."""
    
    def __init__(self, label_smoothing=0.1, entropy_lambda=0.05):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.entropy_lambda = entropy_lambda
        
    def forward(self, logits, labels):
        loss = self.ce_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        if self.entropy_lambda > 0:
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = loss - self.entropy_lambda * entropy
            
        return loss


# ============================================================
# CELL 5: Learning Rate Schedule
# ============================================================

def get_cosine_schedule(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# CELL 6: Data Loading
# ============================================================

# === DATASET PATH ===
# Upload aura_full_dataset.pt to Google Drive, then set this path:
AURA_DATASET_PATH = '/content/drive/MyDrive/aura_full_dataset.pt'


def load_aura_dataset(dataset_path, config, device):
    """
    Load pre-tokenized Aura dataset from aura_full_dataset.pt
    
    Upload aura_full_dataset.pt to Google Drive first, then mount drive:
        from google.colab import drive
        drive.mount('/content/drive')
    
    Args:
        dataset_path: Path to aura_full_dataset.pt
        config: Training config
        device: Target device
    
    Returns:
        List of (input_ids, labels) batches
    """
    print(f"Loading Aura dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    
    sequences = data['sequences']
    print(f"Loaded {len(sequences):,} sequences")
    print(f"Sources: {data.get('sources', 'N/A')}")
    print(f"Total tokens: {data.get('total_tokens', len(sequences) * sequences.shape[1]):,}")
    
    # Create batches
    batches = []
    batch_size = config.batch_size
    
    # Shuffle sequences
    perm = torch.randperm(len(sequences))
    sequences = sequences[perm]
    
    for i in range(0, len(sequences) - batch_size, batch_size):
        batch = sequences[i:i+batch_size].to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        batches.append((input_ids, labels))
    
    print(f"Created {len(batches):,} batches of size {batch_size}")
    return batches


def load_wikitext2(tokenizer, config, device):
    """Load WikiText-2 dataset (fallback if aura_full_dataset.pt not available)."""
    from datasets import load_dataset
    
    print("Loading WikiText-2 dataset (fallback)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Tokenize all texts
    all_tokens = []
    for sample in tqdm(dataset, desc="Tokenizing"):
        text = sample['text']
        if text and len(text.strip()) > 20:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Create sequences
    seq_len = config.max_seq_len
    batch_size = config.batch_size
    num_sequences = len(all_tokens) // seq_len
    sequences = torch.tensor(all_tokens[:num_sequences * seq_len]).reshape(num_sequences, seq_len)
    
    # Shuffle
    perm = torch.randperm(len(sequences))
    sequences = sequences[perm]
    
    # Create batches
    batches = []
    for i in range(0, len(sequences) - batch_size, batch_size):
        batch = sequences[i:i+batch_size].to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        batches.append((input_ids, labels))
    
    print(f"Created {len(batches)} batches")
    return batches


def load_data(config, device, use_aura_dataset=True):
    """
    Load training data.
    
    Args:
        config: Training config
        device: Target device
        use_aura_dataset: If True, load aura_full_dataset.pt; else use WikiText-2
    
    Returns:
        List of (input_ids, labels) batches
    """
    if use_aura_dataset:
        try:
            return load_aura_dataset(AURA_DATASET_PATH, config, device)
        except FileNotFoundError:
            print(f"WARNING: {AURA_DATASET_PATH} not found, falling back to WikiText-2")
    
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
    return load_wikitext2(tokenizer, config, device)


# ============================================================
# CELL 7: Training Loop
# ============================================================

def train(config, model, optimizer, scheduler, criterion, batches, device, checkpoint_dir=None, amygdala: Optional[Amygdala] = None, endocrine: Optional[EndocrineSystem] = None, thalamus: Optional[Thalamus] = None):
    """Main training loop."""
    
    scaler = GradScaler() if config.use_mixed_precision else None
    model.train()
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    memory_gate_scale = 1.0
    
    global_step = 0
    epoch = 0
    losses = []
    best_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    while global_step < config.max_steps:
        epoch += 1
        
        for input_ids, labels in batches:
            if global_step >= config.max_steps:
                break
                
            # Prosody from amygdala (optional)
            prosody = None
            if amygdala is not None:
                prosody = build_prosody(amygdala, model, input_ids)
                limbic_arousal = prosody[..., 0:1].mean().item()
            else:
                limbic_arousal = 0.0

            # Thalamic sensory gating (optional): derive a gating scalar from token embeddings
            thalamus_scale = 1.0
            if thalamus is not None:
                with torch.no_grad():
                    token_embeds = model.semantic_encoder.token_embedding(input_ids)
                    routed, _ = thalamus(token_embeds, limbic_state={'arousal': limbic_arousal})
                    lang = routed.get('language')
                    if lang is not None:
                        thalamus_scale = float(lang.abs().mean().clamp(0.5, 1.5).item())

            # Determine if memory should be active (modulated by endocrine + thalamus)
            base_memory_on = global_step >= config.memory_warmup_steps
            use_memory = base_memory_on and (memory_gate_scale * thalamus_scale >= 0.9)
            store_memory = use_memory and (global_step % 10 == 0)

            # Forward pass
            if config.use_mixed_precision:
                with autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(
                        input_ids, 
                        prosody=prosody, 
                        use_memory=use_memory, 
                        store_memory=store_memory
                    )
                    loss = criterion(logits, labels)
                    loss = loss / config.gradient_accumulation
                    
                scaler.scale(loss).backward()
            else:
                logits, _ = model(
                    input_ids, 
                    prosody=prosody, 
                    use_memory=use_memory, 
                    store_memory=store_memory
                )
                loss = criterion(logits, labels)
                loss = loss / config.gradient_accumulation
                loss.backward()
            
            # Gradient accumulation
            if (global_step + 1) % config.gradient_accumulation == 0:
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                if config.use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            current_loss = loss.item() * config.gradient_accumulation
            losses.append(current_loss)
            
            # Endocrine modulation (optional)
            if endocrine is not None:
                # Simple metrics proxy: accuracy ~ exp(-loss), diversity stub, energy stub
                acc_proxy = float(torch.exp(-loss.detach() * config.gradient_accumulation))
                metrics = {
                    'accuracy': max(0.0, min(1.0, acc_proxy)),
                    'gate_diversity': 0.5,
                    'energy': 0.1,
                }
                hormone_levels = endocrine.step(metrics)
                cortisol = hormone_levels.get('cortisol', 0.0)
                dopamine = hormone_levels.get('dopamine', 0.0)
                norepi = hormone_levels.get('norepinephrine', 0.0)
                thyroid = hormone_levels.get('thyroid', 0.0)
                # Light LR modulation from thyroid/dopamine/cortisol
                lr_scale = 1.0 + 0.01 * (dopamine - cortisol + 0.5 * thyroid)
                lr_scale = float(max(0.9, min(1.1, lr_scale)))
                for base_lr, pg in zip(base_lrs, optimizer.param_groups):
                    pg['lr'] = base_lr * lr_scale
                # Memory gating: boost with norepi/arousal, dampen with cortisol
                memory_gate_scale = float(max(0.8, min(1.2, 1.0 + 0.2 * norepi - 0.2 * cortisol)))
            else:
                hormone_levels = {}
                memory_gate_scale = 1.0

            if global_step % config.eval_interval == 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else current_loss
                lr = scheduler.get_last_lr()[0]
                mem_status = "ON" if use_memory else "OFF"
                mem_count = model.hippocampus.memory_count if model.hippocampus else 0
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{math.exp(min(avg_loss, 20)):.1f}',
                    'lr': f'{lr:.2e}',
                    'mem': f'{mem_status}({mem_count})'
                })
                
                # Memory decay
                if model.hippocampus:
                    # Backward compatibility: older hippocampal versions expose decay_memories only
                    if hasattr(model.hippocampus, "decay"):
                        model.hippocampus.decay(rate=0.001)
                    elif hasattr(model.hippocampus, "decay_memories"):
                        model.hippocampus.decay_memories(decay_rate=0.001)
            
            # Checkpointing
            if checkpoint_dir and global_step > 0 and global_step % config.save_interval == 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(model, optimizer, scheduler, global_step, avg_loss, checkpoint_dir)
            
            global_step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Final checkpoint
    if checkpoint_dir:
        save_checkpoint(model, optimizer, scheduler, global_step, losses[-1], checkpoint_dir, final=True)
    
    return losses


def save_checkpoint(model, optimizer, scheduler, step, loss, checkpoint_dir, final=False):
    """Save training checkpoint."""
    filename = 'checkpoint_final.pt' if final else f'checkpoint_step_{step}.pt'
    path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)
    
    print(f"\nSaved checkpoint: {path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['loss']


# ============================================================
# CELL 8: Main Execution
# ============================================================

def main(config_preset='test'):
    """
    Main training function.
    
    Args:
        config_preset: 'test' (5K steps), 'medium' (20K steps), or 'full' (50K steps)
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Config
    if config_preset == 'baseline':
        config = get_baseline_config()
        print("\nUsing BASELINE config (no SNN/RAG - for debugging)")
    elif config_preset == 'medium':
        config = get_medium_config()
        print("\nUsing MEDIUM config (20K steps, ~2 hours)")
    elif config_preset == 'full':
        config = get_full_config()
        print("\nUsing FULL config (50K steps, ~6 hours)")
    else:
        config = get_test_config()
        print("\nUsing TEST config (5K steps, ~15 min)")
    print(f"\nModel: {config.embedding_dim}D x {config.num_layers}L x {config.num_heads}H")
    print(f"SNN layers: {config.snn_layers}")
    print(f"RAG: {config.use_rag}, Memory injection: {config.memory_injection}")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"Max steps: {config.max_steps}")
    
    # Tokenizer
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")
    
    # Data - Set use_aura_dataset=True to use aura_full_dataset.pt
    # Make sure to upload aura_full_dataset.pt to Google Drive first!
    batches = load_data(config, device, use_aura_dataset=True)
    
    # Model
    hippocampus = HippocampalFormation(
        feature_dim=config.embedding_dim,
        n_place_cells=config.n_place_cells,
        n_time_cells=config.n_time_cells,
        n_grid_cells=config.n_grid_cells,
        max_memories=config.max_memories,
        device=str(device),
        use_centroid_index=config.enable_centroid_index
    ).to(device)
    
    model = HippocampalTransformer(config, hippocampus).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optional: load pretrained checkpoint
    start_step = 0
    if config.checkpoint_path:
        ckpt_path = Path(config.checkpoint_path)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
                print(f"Loaded checkpoint from {ckpt_path}")
                if config.load_optimizer:
                    if 'optimizer_state_dict' in ckpt:
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=config.lr,
                            weight_decay=config.weight_decay
                        )
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    if 'scheduler_state_dict' in ckpt:
                        scheduler = get_cosine_schedule(optimizer, config.warmup_steps, config.max_steps, config.min_lr_ratio)
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    start_step = ckpt.get('step', 0)
                    if start_step > 0:
                        config.max_steps = start_step + config.max_steps
                        print(f"Resuming from step {start_step}, new max_steps={config.max_steps}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

    # Optional: build limbic/endocrine/thalamus modulators
    amygdala = Amygdala(config.embedding_dim).to(device) if config.enable_amygdala else None
    endocrine = EndocrineSystem() if config.enable_endocrine else None
    thalamus = Thalamus(d_model=config.embedding_dim, region_names=['language'], top_k=1).to(device) if config.enable_thalamus else None

    # Optional: build continuous learning orchestrator (memory-only mode)
    cl_orchestrator = build_continuous_learning_orchestrator(
        config, 
        hippocampus, 
        tokenizer=tokenizer, 
        model=model, 
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule(optimizer, config.warmup_steps, config.max_steps, config.min_lr_ratio)
    criterion = AuraLoss(label_smoothing=config.label_smoothing, entropy_lambda=config.entropy_lambda)
    
    # Checkpoint directory (for Colab with Drive)
    checkpoint_dir = None
    # Uncomment for Google Drive:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # checkpoint_dir = '/content/drive/MyDrive/aura_checkpoints'
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train
    losses = train(
        config, 
        model, 
        optimizer, 
        scheduler, 
        criterion, 
        batches, 
        device, 
        checkpoint_dir, 
        amygdala=amygdala, 
        endocrine=endocrine,
        thalamus=thalamus
    )
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial loss: {sum(losses[:10])/10:.4f}")
    print(f"Final loss: {sum(losses[-10:])/10:.4f}")
    print(f"Final perplexity: {math.exp(sum(losses[-10:])/10):.2f}")
    print(f"Memories stored: {hippocampus.memory_count}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    prompts = [
        "The meaning of life is",
        "In the beginning",
        "Scientists have discovered that",
        "The most important thing about"
    ]
    
    def sample_generate(prompt_text: str, max_new_tokens: int = 40):
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=device)
        generated = prompt_ids
        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = generated[:, -config.max_seq_len:] if generated.shape[1] > config.max_seq_len else generated
                logits, _ = model(context, use_memory=True, store_memory=False)
                next_logits = logits[:, -1, :]
                next_logits = next_logits / 0.7  # temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                    break
        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    for prompt_text in prompts:
        generated_text = sample_generate(prompt_text, max_new_tokens=40)
        print(f"\nPrompt: {prompt_text}")
        print(f"Output: {generated_text}")
    
    # Training status
    print("\n" + "=" * 60)
    print("TRAINING STATUS")
    print("=" * 60)
    final_ppl = math.exp(sum(losses[-10:])/10)
    if final_ppl < 100:
        print("Good progress! Model is learning.")
        if final_ppl < 50:
            print("Consider running with get_medium_config() for better results.")
        if final_ppl < 20:
            print("Excellent! Ready for get_full_config() training.")
    else:
        print("Model needs more training. Current output may be incoherent.")
        print("This is expected for short runs. Increase max_steps for better results.")
    
    return model, losses, cl_orchestrator


if __name__ == '__main__':
    # Change to 'medium' or 'full' for longer training
    # 'test': 5K steps, ~15 min, PPL ~100-200
    # 'medium': 20K steps, ~2 hours, PPL ~30-50
    # 'full': 50K steps, ~6 hours, PPL ~10-20
    
    model, losses, cl_orchestrator = main(config_preset='test')
    
    # To continue training with a larger config:
    # model, losses, cl_orchestrator = main(config_preset='medium')

