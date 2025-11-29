import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.core.hippocampal import HippocampalFormation


def test_centroid_index_biases_retrieval():
    """Centroid index should bias retrieval toward the nearest cluster."""
    hf = HippocampalFormation(
        n_place_cells=10,
        n_time_cells=5,
        n_grid_cells=5,
        max_memories=100,
        feature_dim=4,
        device="cpu",
    )
    # Use a small centroid count and force early rebuilds for the test
    hf.centroids_k = 4
    hf.centroids_update_interval = 1

    # Cluster A near [1, 0, 0, 0], Cluster B near [0, 1, 0, 0]
    for i in range(10):
        hf.create_episodic_memory(
            memory_id=f"A{i}",
            event_id=f"A{i}",
            features=torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
        )
    for i in range(10):
        hf.create_episodic_memory(
            memory_id=f"B{i}",
            event_id=f"B{i}",
            features=torch.tensor([0.0, 1.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
        )

    # Build the centroid index
    hf.rebuild_centroids()
    assert hf._index_ready is True
    assert hf.memory_count == 20

    # Query near cluster A and ensure retrieved ids are from A
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    results = hf.retrieve_similar_memories(query, k=5)
    retrieved_ids = {rid for rid, _ in results}
    assert len(results) == 5
    assert all(rid.startswith("A") for rid in retrieved_ids)


def test_retrieval_fallback_when_small_bank():
    """When memory_count < centroids_k, retrieval should still work (fallback)."""
    hf = HippocampalFormation(
        n_place_cells=10,
        n_time_cells=5,
        n_grid_cells=5,
        max_memories=50,
        feature_dim=4,
        device="cpu",
    )
    # Very small bank, no centroid index ready
    for i in range(3):
        hf.create_episodic_memory(
            memory_id=f"S{i}",
            event_id=f"S{i}",
            features=torch.tensor([float(i == 0), float(i == 1), 0.0, 0.0]),
        )
    assert hf.memory_count == 3
    assert hf._index_ready is False
    results = hf.retrieve_similar_memories(torch.tensor([1.0, 0.0, 0.0, 0.0]), k=2)
    assert len(results) == 2


def test_decay_memories_reduces_strength():
    hf = HippocampalFormation(
        n_place_cells=5,
        n_time_cells=5,
        n_grid_cells=5,
        max_memories=10,
        feature_dim=4,
        device="cpu",
    )
    hf.create_episodic_memory("X", "X", torch.zeros(4))
    before = hf.memory_metadata[0, 0].item()
    hf.decay_memories(decay_rate=0.1)
    after = hf.memory_metadata[0, 0].item()
    assert after < before
    assert after > 0.0
