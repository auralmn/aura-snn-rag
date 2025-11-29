# Aura Project Overview

Aura is a generative hybrid ANN–SNN hippocampal transformer that blends sparse spiking components, episodic memory, and limbic/endocrine modulation to push language models toward human-like efficiency and continual learning.

## What’s in scope
- Hippocampal memory with place/grid/time cells, one-shot episodic writes, and centroid-based retrieval for sublinear lookup
- Limbic/endocrine/thalamic gates (amygdala for affect/prosody, endocrine for LR/memory scaling, thalamus for sensory gating)
- Optional spiking FFN blocks and continuous-learning orchestrator with replay/ingestion pipelines
- Prosody and identity signals via an emotion/intent/tone/personality head (trainable on vocab_src corpora)
- Mixture-of-depths roadmap: dynamic layer execution by token salience (design in progress)

## Repo layout
- `src/core`: hippocampal transformer, limbic/endocrine/thalamus modules, language_zone/*
- `src/prosody/emotion_head.py`: multitask head scaffolding (emotion/intent/tone/personality)
- `scripts`: utilities (`run_tests.py`, `train_emotion_head.py`, dataset prep helpers)
- `models`: checkpoints (default `models/checkpoint_final.pt`, 50k steps; ~271M params)
- `vocab_src`: primary training corpora (~253M tokens; FLAN-T5 tokenizer)
- `docs`: `ARCHITECTURE.md` with mermaid diagrams; training/optimization notes
- `tests`: curated unit/integration set; legacy-heavy files are skipped by markers

## Quickstart
```bash
pip install -r requirements.txt   # or uv sync
python repo/scripts/run_tests.py  # curated suite; legacy tests skipped
```

```python
from colab_l4_training import get_test_config, main
cfg = get_test_config()
cfg.checkpoint_path = "models/checkpoint_final.pt"  # autoload weights
cfg.load_optimizer = False  # set True to resume LR/scheduler for more steps
model, losses, cl_orch = main(config_preset="test")
```

## Training and continuation
- Default autoload: `models/checkpoint_final.pt` (50k steps). Set `load_optimizer=True` to resume optimizer/scheduler and extend `max_steps` for another 50k (toward 100k).
- Config toggles: `enable_amygdala`, `enable_endocrine`, `enable_thalamus`, `enable_centroid_index`, `enable_continuous_learning`, `load_optimizer`, `checkpoint_path`.
- One-shot memory + generation helpers: `ingest_jsonl_to_memory`, `ingest_csv_pairs_to_memory`, `one_shot_memorize_and_generate`.
- Centroid index: lightweight approximate nearest neighbors to reduce the O(n²) cognitive map cost; can be rebuilt incrementally.

## Emotion/identity head (optional)
- Train with vocab_src corpora (e.g., `amygdala_affect.jsonl`, `emotion_valence_arousal_realm_phase.jsonl`, `identitya.jsonl`):
```bash
python repo/scripts/train_emotion_head.py --epochs 3 --max-examples 5000
```
- Outputs to `models/emotion_head/emotion_head.pt` with label maps; wiring to full prosody pipeline is scaffolded and ready for integration.

## Data plan
- Current: ~253M tokens from `vocab_src` (no WikiText). Specialty sets include principles, timeline Q&A, intents, greetings, OpenThoughts, and more.
- Next: Nemotron CC v2 (high-quality + code) to raise quality from moderate→high in later phases.

## Notebooks and phases
- `notebooks/phase1_core_lm.ipynb`: core LM training
- `phase2_memory_seeding.ipynb`: episodic ingestion and one-shot recall
- `phase3_specialists.ipynb`: specialist creation/adaptation by topic
- `phase4_continuous_learning.ipynb`: orchestrator, ingestion, ablations

## Testing and benchmarks
- Run curated suite: `python repo/scripts/run_tests.py`
- Coverage: centroid index, modulation (amygdala/endocrine/thalamus), ingestion, emotion head
- Bench/ablation toggles live in configs; respect empirical proof process before promoting changes.

## Current risks and mitigations
- Cognitive map O(n²): mitigated by centroid index; explore graph/ANN backends next (Pinecone/Qdrant later).
- Replay buffer sizing and decay: add learned decay and dynamic buffer in upcoming experiments.
- Keep artifacts >2GB out of git/LFS to avoid push failures; store checkpoints locally.

## License
MIT
