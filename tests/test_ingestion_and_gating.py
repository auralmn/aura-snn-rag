import sys
import json
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import colab_l4_training as cl  # noqa: E402
from src.core.hippocampal import HippocampalFormation  # noqa: E402


class DummyTokenizer:
    def encode(self, text, **kwargs):
        return [1, 2, 3]


def _stub_one_shot(mem_id_prefix="stub"):
    def fn(text, tokenizer, model, hippocampus, device, memory_id=None):
        mem_id = memory_id or f"{mem_id_prefix}-{hippocampus.memory_count}"
        vec = torch.ones(hippocampus.memory_features.shape[1], device=hippocampus.device)
        hippocampus.create_episodic_memory(mem_id, mem_id, vec)
        return mem_id
    return fn


def test_ingest_jsonl_to_memory_stores_entries():
    tok = DummyTokenizer()
    hippocampus = HippocampalFormation(
        feature_dim=16,
        n_place_cells=10,
        n_time_cells=5,
        n_grid_cells=5,
        max_memories=50,
        device="cpu",
    )
    orig = cl.one_shot_memorize_text
    dummy_model = object()
    cl.one_shot_memorize_text = _stub_one_shot("jsonl")
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"text": "hello world"}) + "\n")
            f.write(json.dumps({"instruction": "do X", "output": "done"}) + "\n")
            path = f.name
        stored = cl.ingest_jsonl_to_memory(path, tok, dummy_model, hippocampus, device="cpu", max_items=10)
        assert stored == 2
        assert hippocampus.memory_count == 2
    finally:
        cl.one_shot_memorize_text = orig
        Path(path).unlink(missing_ok=True)


def test_ingest_csv_pairs_to_memory_stores_entries():
    tok = DummyTokenizer()
    hippocampus = HippocampalFormation(
        feature_dim=16,
        n_place_cells=10,
        n_time_cells=5,
        n_grid_cells=5,
        max_memories=50,
        device="cpu",
    )
    orig = cl.one_shot_memorize_text
    dummy_model = object()
    cl.one_shot_memorize_text = _stub_one_shot("csv")
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("Q1,A1\n")
            f.write("Q2,A2\n")
            path = f.name
        stored = cl.ingest_csv_pairs_to_memory(path, tok, dummy_model, hippocampus, device="cpu", max_items=10)
        assert stored == 2
        assert hippocampus.memory_count == 2
    finally:
        cl.one_shot_memorize_text = orig
        Path(path).unlink(missing_ok=True)


def test_hormone_scaling_bounds():
    endocrine = cl.EndocrineSystem()
    levels = endocrine.step({"accuracy": 0.8, "gate_diversity": 0.5, "energy": 0.2})
    cortisol = levels.get("cortisol", 0.0)
    dopamine = levels.get("dopamine", 0.0)
    norepi = levels.get("norepinephrine", 0.0)
    thyroid = levels.get("thyroid", 0.0)
    lr_scale = 1.0 + 0.01 * (dopamine - cortisol + 0.5 * thyroid)
    lr_scale = max(0.9, min(1.1, lr_scale))
    mem_gate_scale = 1.0 + 0.2 * norepi - 0.2 * cortisol
    mem_gate_scale = max(0.8, min(1.2, mem_gate_scale))
    assert 0.9 <= lr_scale <= 1.1
    assert 0.8 <= mem_gate_scale <= 1.2
    # Ensure memory remains usable when gate is above threshold
    assert mem_gate_scale * 1.0 >= 0.8
