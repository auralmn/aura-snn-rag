---
language:
- en
license: mit
library_name: pytorch
tags:
- neuromorphic
- hippocampal-transformer
- bio-inspired
- episodic-memory
- spiking-neural-networks
- hybrid-learning
- place-cells
- grid-cells
- time-cells
base_model: google/flan-t5-base
pipeline_tag: text-generation
---

# Model Card — Aura SNN-RAG (HippocampalTransformer v0.2)

A neuromorphic, hybrid ANN–SNN language model that couples a hippocampal episodic memory system with a transformer backbone, limbic/endocrine modulators, and optional spiking layers. Goal: energy-aware, memory-centric language modeling with continual learning.

## Model Details

**Developed by:** Aura Team (auralmn)  
**Languages:** English  
**License:** MIT  
**Base tokenizer:** google/flan-t5-base  
**Parameters:** ~271M (transformer + hippocampal components)  
**Checkpoint:** `models/checkpoint_final.pt` (50k steps)  
**Repo:** https://github.com/auralmn/aura-snn-rag

### Architecture
- Transformer decoder with hippocampal gating (place/grid/time cells) and one-shot episodic writes
- Centroid-based approximate memory index to reduce cognitive-map O(n²) cost
- Limbic/endocrine/thalamic modulators: affect-prosody signals, LR/memory scaling, sensory gating
- Optional spiking FFN blocks; prosody/identity head scaffold (emotion/intent/tone/personality)
- Continuous learning orchestrator with replay/ingestion hooks

## Intended Use
- Research on neuromorphic/biologically inspired language modeling
- Memory-augmented generation, one-shot episodic recall, continual-learning experiments
- Benchmarking ablations (amygdala/endocrine/thalamus/centroid_index toggles)

### Out of scope
- Safety-critical or production deployments
- Factual tasks without verification (model may confabulate)
- Low-latency serving without further optimization

### Model Sources

- **Repository:** https://github.com/auralmn/aura-snn-rag
- **Docs:** `docs/ARCHITECTURE.md` (mermaid diagrams), `README.md`
- **Checkpoint:** `models/checkpoint_final.pt` (local; >2GB, keep out of git/LFS)

## Bias, Risks, and Limitations

- Centroid index is experimental; verify retrieval quality vs speed.
- Memory decay/replay buffers are hand-tuned; risk of forgetting or over-retention.
- Prosody/identity head is scaffold-only unless trained weights are provided.
- May confabulate; inherits biases from training corpora.

## How to Get Started with the Model

```python
import torch
from colab_l4_training import get_test_config, main

cfg = get_test_config()
cfg.checkpoint_path = "models/checkpoint_final.pt"
cfg.load_optimizer = False      # set True to resume optimizer/scheduler for continuation
cfg.enable_centroid_index = True
cfg.enable_amygdala = True
cfg.enable_endocrine = True

model, losses, cl_orch = main(config_preset="test")
model.eval()
prompt = "Describe how episodic memory can aid question answering."
# ... tokenize prompt with the FLAN-T5 tokenizer and generate ...
```

## Training Details

**Status:** 50k steps on ~253M tokens (vocab_src corpora). Goal: continue +50k steps (load optimizer state optional).  
**Data:** curated JSONL/CSV corpora in `vocab_src` (principles, timeline Q&A, intents, greetings, OpenThoughts, affect/identity sets); no WikiText. Next phase: Nemotron CC v2 high-quality + code.  
**Objective:** causal LM with memory gating; optional continuous learning + ingestion hooks.  
**Hardware:** L4-class GPU (Colab) for the current checkpoint.

## Evaluation

Current automated coverage (curated suite, legacy heavy tests skipped):
- Centroid index retrieval and ingestion
- Modulation (amygdala/endocrine/thalamus) wiring
- Emotion head forward/loss
- Ingestion + gating paths
Use `python repo/scripts/run_tests.py` for the curated suite. Benchmarks/ablations are WIP.

## Environmental Impact
- Training to 50k steps on L4-class GPU (Colab) — moderate compute budget; exact CO2 not audited. Optimize further by enabling centroid index and pruning memory rebuilds.

## Citation

```bibtex
@software{aura_snn_rag_2025,
  title={Aura SNN-RAG: Hybrid Hippocampal Transformer with Spiking Components},
  author={Aura Team},
  year={2025},
  url={https://github.com/auralmn/aura-snn-rag},
  note={Checkpoint 50k steps, ~271M params}
}
```
