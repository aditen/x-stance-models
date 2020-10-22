from __future__ import annotations

from dataclasses import dataclass
from typing import List

from models.properties.model_properties import ModelProperties


@dataclass
class AttentionModelProperties(ModelProperties):
    masking_strategies: List[str]
    num_layers: int = 2
    num_heads: int = 2
    hidden_size: int = 128
    intermediate_size: int = hidden_size * 2
    dropout: float = 0.25
    activation_function: str = "gelu"

    @staticmethod
    def small_bertolt() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="bertolt_small", initial_lr=1e-4, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["question_only", "comment_only", "pad"], hidden_size=128,
            intermediate_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            num_layers=3, num_heads=2,
            dropout=0.15, activation_function="gelu")

    @staticmethod
    def large_bertolt() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="bertolt_large", initial_lr=5e-5, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["question_only", "comment_only", "question_only", "comment_only", "pad"],
            hidden_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            intermediate_size=512,
            num_layers=5, num_heads=4,
            dropout=0.3, activation_function="gelu")

    @staticmethod
    def small_bertrand() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="bertrand_small", initial_lr=1e-4, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["comment_only", "comment_only", "pad"], hidden_size=128,
            intermediate_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            num_layers=3, num_heads=2,
            dropout=0.15, activation_function="gelu")

    @staticmethod
    def large_bertrand() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="bertrand_large", initial_lr=5e-5, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["comment_only", "comment_only", "comment_only", "comment_only", "pad"],
            hidden_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            intermediate_size=512,
            num_layers=5, num_heads=4,
            dropout=0.3, activation_function="gelu")

    @staticmethod
    def small_mask_baseline() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="mask_baseline_small", initial_lr=1e-4, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["pad", "pad", "pad"], hidden_size=128,
            intermediate_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            num_layers=3, num_heads=2,
            dropout=0.15, activation_function="gelu")

    @staticmethod
    def large_mask_baseline() -> AttentionModelProperties:
        return AttentionModelProperties(
            model_name="mask_baseline_large", initial_lr=5e-5, step_size=10, gamma=0.75, batch_size=16,
            masking_strategies=["pad", "pad", "pad", "pad", "pad"],
            hidden_size=256, n_epochs=25, embedding=None, bp_emb_size=None,
            intermediate_size=512,
            num_layers=5, num_heads=4,
            dropout=0.3, activation_function="gelu")

    @staticmethod
    def small_ordinary() -> AttentionModelProperties:
        return AttentionModelProperties(model_name="custom_transformer_small", initial_lr=1e-4, step_size=10,
                                        gamma=0.75, batch_size=16, n_epochs=25, embedding=None, bp_emb_size=None,
                                        masking_strategies=["pad", "pad"], hidden_size=128, intermediate_size=256,
                                        num_layers=2, num_heads=2,
                                        dropout=0.15, activation_function="gelu")

    @staticmethod
    def medium_ordinary() -> AttentionModelProperties:
        return AttentionModelProperties(model_name="custom_transformer_medium", initial_lr=5e-5, step_size=10,
                                        gamma=0.75, batch_size=16, n_epochs=25, embedding=None, bp_emb_size=None,
                                        masking_strategies=["pad", "pad", "pad", "pad"],
                                        hidden_size=256, intermediate_size=512,
                                        num_layers=4, num_heads=4,
                                        dropout=0.3, activation_function="gelu")

    @staticmethod
    def large_ordinary() -> AttentionModelProperties:
        return AttentionModelProperties(model_name="custom_transformer_large", initial_lr=5e-5, step_size=10,
                                        gamma=0.75, batch_size=16, n_epochs=25, embedding=None, bp_emb_size=None,
                                        masking_strategies=["pad", "pad", "pad", "pad"],
                                        hidden_size=512, intermediate_size=1024,
                                        num_layers=4, num_heads=8,
                                        dropout=0.3, activation_function="gelu")
