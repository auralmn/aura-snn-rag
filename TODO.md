# Aura SNN-RAG — Near-Term TODO

Context: repo is now the git root pushed to `auralmn/aura-snn-rag`. Large artifacts (>2GB) stay local (models/, data dumps).

## Wiring and inference
- [ ] Integrate emotion/identity head into prosody pipeline with a config flag; add smoke test.
- [ ] Thalamus + endocrine + amygdala: add end-to-end gating test that spans ingestion → retrieval → generation.
- [ ] Specialist auto-creation: surface a minimal API/notebook to spawn specialists per topic with phase-separated notebooks.

## Memory/index
- [ ] Centroid index: expose rebuild schedule + persistence; add option to swap in ANN backend later (Pinecone/Qdrant adapter).
- [ ] Cognitive map O(n²): experiment with sparse graph construction (k-NN edges) and verify retrieval quality vs speed.
- [ ] Replay/decay: make decay learnable and buffer size adaptive; add ablation toggles in config and tests.

## Training/bench/ablations
- [ ] Bench runner: add scripts to run curated benchmarks + ablations (amygdala/endocrine/thalamus/centroid_index toggles).
- [ ] Resume path: document continuing from `models/checkpoint_final.pt` for +50k steps (optimizer state optional).
- [ ] Emotion head: finish training recipe on vocab_src; optionally pretrain personality head (identitya.jsonl).

## Legacy reuse
- [ ] Pull useful pieces from `old/training/*` (e.g., amygdala trainer) where it adds value; discard if redundant.
- [ ] Keep personality/head room for future identity/mood conditioning (no-op config until trained weights exist).

## Docs
- [ ] Keep `ARCHITECTURE.md` and README in sync with current toggles, checkpoints, and data sources.
- [ ] Add short guide for pushing without LFS bloat (ignore checkpoints >2GB).
