from models.pipelines.model.bow_pipeline import BowPipeline
from models.pipelines.model.self_attention_pipeline import AttentionPipeline
from models.properties.attention_model_properties import AttentionModelProperties
from models.properties.bow_properties import BoWModelProperties

bow_embedding_sizes = False
bow_ngram_sizes = False
bow_pre_trained_embeddings = True

self_attention_sizes = False
self_attention_masks = False

if __name__ == "__main__":

    if bow_embedding_sizes:
        BowPipeline(BoWModelProperties.tiny_own_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.small_own_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.medium_own_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.large_own_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.huge_own_embedding()).train_and_validate_for_epochs_and_test()

    if bow_ngram_sizes:
        BowPipeline(BoWModelProperties.bigram()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.trigram()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.fourgram()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.fivegram()).train_and_validate_for_epochs_and_test()

    if bow_pre_trained_embeddings:
        BowPipeline(BoWModelProperties.small_bp_emb_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.medium_bp_emb_embedding()).train_and_validate_for_epochs_and_test()
        BowPipeline(BoWModelProperties.fasttext_embedding()).train_and_validate_for_epochs_and_test()

    if self_attention_sizes:
        AttentionPipeline(AttentionModelProperties.small_ordinary()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.medium_ordinary()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.large_ordinary()).train_and_validate_for_epochs_and_test()

    if self_attention_masks:
        AttentionPipeline(AttentionModelProperties.small_mask_baseline()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.large_mask_baseline()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.small_bertolt()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.large_bertolt()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.small_bertrand()).train_and_validate_for_epochs_and_test()
        AttentionPipeline(AttentionModelProperties.large_bertrand()).train_and_validate_for_epochs_and_test()
