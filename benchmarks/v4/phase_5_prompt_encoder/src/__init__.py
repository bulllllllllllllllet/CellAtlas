"""Phase-5 prompt data and model components."""

from .prompts import (
    PROMPT_SIZE_SPECS,
    PromptEpisode,
    dominant_parent_slots,
    hard_region_adjacency,
    sample_connected_region_set,
    sample_region_prompt_episode,
)

__all__ = [
    "PROMPT_SIZE_SPECS",
    "PromptEpisode",
    "dominant_parent_slots",
    "hard_region_adjacency",
    "sample_connected_region_set",
    "sample_region_prompt_episode",
]
