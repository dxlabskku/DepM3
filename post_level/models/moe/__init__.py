"""Multimodal Mixture of Experts modules."""

from .multimodal_moe import (
    ModalityExpert,
    ModalityAwareGating,
    MultimodalMoE,
    MultimodalMoEBlock,
    # Pre-CoSSM FiLM-based Modality-aware MoE
    FiLMExpert,
    FiLMModalityMoE,
)

__all__ = [
    'ModalityExpert',
    'ModalityAwareGating', 
    'MultimodalMoE',
    'MultimodalMoEBlock',
    # Pre-CoSSM FiLM MoE
    'FiLMExpert',
    'FiLMModalityMoE',
]

