"""Multimodal sensor fusion module (Phase 5-6: deferred encoders to Phase 9-10)."""

from src.fusion.fusion_model import SensorFusionProcessor, create_fusion_processor

__all__ = [
    'SensorFusionProcessor',
    'create_fusion_processor',
]
