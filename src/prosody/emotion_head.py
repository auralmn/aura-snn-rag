"""
Emotion / Personality head for prosody and identity signals.

This module can be trained on top of pooled token embeddings (e.g., FLAN-T5) to
produce multi-task outputs:
 - emotion
 - intent
 - tone
 - personality/mood

At inference time, the head can be used to generate prosody cues (arousal/valence)
or to condition downstream modules on identity/mood.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class EmotionPersonalityHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_emotions: int,
        num_intents: int,
        num_tones: int,
        num_personalities: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.tone_head = nn.Linear(hidden_dim, num_tones)
        self.personality_head = nn.Linear(hidden_dim, num_personalities)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooled: [batch, embed_dim] pooled token embeddings
        Returns:
            dict of logits for each task
        """
        h = self.shared(pooled)
        return {
            "emotion": self.emotion_head(h),
            "intent": self.intent_head(h),
            "tone": self.tone_head(h),
            "personality": self.personality_head(h),
        }


class EmotionPersonalityLoss(nn.Module):
    def __init__(
        self,
        w_emotion: float = 1.0,
        w_intent: float = 0.7,
        w_tone: float = 0.7,
        w_personality: float = 0.7,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.w_emotion = w_emotion
        self.w_intent = w_intent
        self.w_tone = w_tone
        self.w_personality = w_personality

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        if "emotion" in logits and "emotion" in targets:
            loss = loss + self.w_emotion * self.ce(logits["emotion"], targets["emotion"])
        if "intent" in logits and "intent" in targets:
            loss = loss + self.w_intent * self.ce(logits["intent"], targets["intent"])
        if "tone" in logits and "tone" in targets:
            loss = loss + self.w_tone * self.ce(logits["tone"], targets["tone"])
        if "personality" in logits and "personality" in targets:
            loss = loss + self.w_personality * self.ce(logits["personality"], targets["personality"])
        return loss


def pool_token_embeddings(hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pool token embeddings.

    Args:
        hidden_states: [batch, seq, dim]
        mask: optional [batch, seq] attention mask (1 for tokens, 0 for padding)
    Returns:
        [batch, dim] pooled embeddings
    """
    if mask is None:
        return hidden_states.mean(dim=1)
    mask = mask.unsqueeze(-1)  # [B, S, 1]
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom
