import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.prosody.emotion_head import EmotionPersonalityHead, EmotionPersonalityLoss, pool_token_embeddings  # noqa: E402


def test_emotion_head_shapes():
    head = EmotionPersonalityHead(
        embed_dim=32,
        num_emotions=4,
        num_intents=3,
        num_tones=2,
        num_personalities=5,
        hidden_dim=16,
        dropout=0.0,
    )
    pooled = torch.randn(2, 32)
    out = head(pooled)
    assert set(out.keys()) == {"emotion", "intent", "tone", "personality"}
    assert out["emotion"].shape == (2, 4)
    assert out["personality"].shape == (2, 5)


def test_emotion_head_loss():
    head = EmotionPersonalityHead(
        embed_dim=8,
        num_emotions=2,
        num_intents=2,
        num_tones=2,
        num_personalities=2,
    )
    crit = EmotionPersonalityLoss()
    pooled = torch.randn(3, 8)
    logits = head(pooled)
    targets = {
        "emotion": torch.tensor([0, 1, 0]),
        "intent": torch.tensor([1, 0, 1]),
        "tone": torch.tensor([0, 0, 1]),
        "personality": torch.tensor([1, 1, 0]),
    }
    loss = crit(logits, targets)
    assert loss > 0


def test_pool_token_embeddings_mask():
    hidden = torch.randn(1, 4, 6)
    mask = torch.tensor([[1, 1, 0, 0]])
    pooled = pool_token_embeddings(hidden, mask)
    assert pooled.shape == (1, 6)
    # masked positions should not contribute (compare to manual mean of first two tokens)
    manual = hidden[:, :2].mean(dim=1)
    assert torch.allclose(pooled, manual)
