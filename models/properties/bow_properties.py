from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.properties.model_properties import ModelProperties


@dataclass
class BoWModelProperties(ModelProperties):
    embedding: Optional[str]
    n_grams: int
    embedding_dim: int
    freeze: bool

    @staticmethod
    def tiny_own_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_own_tiny", initial_lr=2e-4, step_size=10, gamma=0.75, batch_size=16,
                                  n_epochs=30, n_grams=1, embedding_dim=32, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def small_own_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_own_small", initial_lr=2e-4, step_size=10, gamma=0.75, batch_size=16,
                                  n_epochs=30, n_grams=1, embedding_dim=64, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def medium_own_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_own_medium", initial_lr=15e-5, step_size=10, gamma=0.75,
                                  batch_size=16, n_epochs=30, n_grams=1, embedding_dim=128, embedding=None,
                                  bp_emb_size=None, freeze=False)

    @staticmethod
    def large_own_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_own_large", initial_lr=1e-4, step_size=10, gamma=0.75, batch_size=16,
                                  n_epochs=30, n_grams=1, embedding_dim=300, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def huge_own_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_own_huge", initial_lr=1e-4, step_size=10, gamma=0.75, batch_size=16,
                                  n_epochs=30, n_grams=1, embedding_dim=512, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def bigram() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_bigrams", initial_lr=5e-5, step_size=12, gamma=0.75, batch_size=16,
                                  n_epochs=36, n_grams=2, embedding_dim=300, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def trigram() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_trigrams", initial_lr=5e-5, step_size=12, gamma=0.75, batch_size=16,
                                  n_epochs=36, n_grams=3, embedding_dim=300, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def fourgram() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_fourgrams", initial_lr=5e-5, step_size=12, gamma=0.75, batch_size=16,
                                  n_epochs=36, n_grams=4, embedding_dim=300, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def fivegram() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_fivegrams", initial_lr=5e-5, step_size=12, gamma=0.75, batch_size=16,
                                  n_epochs=36, n_grams=5, embedding_dim=300, embedding=None, bp_emb_size=None,
                                  freeze=False)

    @staticmethod
    def small_bp_emb_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_bpemb_s", initial_lr=5e-3, step_size=12, gamma=0.75,
                                  batch_size=16, n_epochs=40, n_grams=1, bp_emb_size=100000, embedding_dim=300,
                                  embedding="bpemb", freeze=True)

    @staticmethod
    def medium_bp_emb_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_bpemb_m", initial_lr=5e-3, step_size=12, gamma=0.75,
                                  batch_size=16, n_epochs=40, n_grams=1, bp_emb_size=320000, embedding_dim=300,
                                  embedding="bpemb", freeze=True)

    @staticmethod
    def fasttext_embedding() -> BoWModelProperties:
        return BoWModelProperties(model_name="bow_fasttext", initial_lr=5e-3, step_size=12, gamma=0.75,
                                  batch_size=16, n_epochs=40, n_grams=1, embedding_dim=300, embedding="fasttext",
                                  bp_emb_size=None, freeze=True)
