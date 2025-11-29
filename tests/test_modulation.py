import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from colab_l4_training import (
    get_test_config,
    build_prosody,
)
from core.hippocampal import HippocampalFormation
from core.language_zone.hippocampal_transformer import HippocampalTransformer
from core.limbic_system import Amygdala
from core.endocrine import EndocrineSystem
from core.thalamus import Thalamus


def _build_model(device):
    config = get_test_config()
    config.use_mixed_precision = False
    hippocampus = HippocampalFormation(
        feature_dim=config.embedding_dim,
        n_place_cells=config.n_place_cells,
        n_time_cells=config.n_time_cells,
        n_grid_cells=config.n_grid_cells,
        max_memories=config.max_memories,
        device=str(device),
    ).to(device)
    model = HippocampalTransformer(config, hippocampus).to(device)
    return config, model, hippocampus


def test_build_prosody_shape_and_values():
    device = torch.device("cpu")
    _, model, _ = _build_model(device)
    amygdala = Amygdala(model.config.embedding_dim).to(device)
    input_ids = torch.randint(0, 50, (2, 8), device=device)
    prosody = build_prosody(amygdala, model, input_ids)
    assert prosody.shape == (2, 8, 4)
    # arousal/valence signals should be finite
    assert torch.isfinite(prosody).all()


def test_thalamus_routes_language():
    device = torch.device("cpu")
    thalamus = Thalamus(d_model=16, region_names=["language"], top_k=1).to(device)
    x = torch.randn(2, 5, 16, device=device)
    routed, stats = thalamus(x, limbic_state={"arousal": 0.5})
    assert "language" in routed
    assert routed["language"].shape == x.shape
    assert isinstance(stats, torch.Tensor)


def test_endocrine_step_returns_hormones():
    endocrine = EndocrineSystem()
    levels = endocrine.step({"accuracy": 0.8, "gate_diversity": 0.5, "energy": 0.2})
    assert "cortisol" in levels
    assert "dopamine" in levels
    for v in levels.values():
        assert v >= 0.0
