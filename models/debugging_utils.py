from models.pipelines.embedding.embedding_pipeline import EmbeddingPipeline
from models.pipelines.model.bow_pipeline import BowPipeline
from models.pipelines.model.model_pipeline import ModelPipeline
from models.pipelines.model.self_attention_pipeline import AttentionPipeline
from models.pipelines.pipeline_utils import PipelineUtils


class XStanceDebuggingUtils:
    @staticmethod
    def print_tokens_based_on_embedding(embedding: EmbeddingPipeline, tokens):
        for tok in tokens:
            print(embedding.itos[tok], end=" ")
        print()

    # If necessary also add a "set all indicies" (but be careful because of the index(<sep>) call!)
    @staticmethod
    def override_3th_index_in_bow_dataset(bow_pipeline: BowPipeline):
        for i in range(len(bow_pipeline.training_data)):
            label, actual = bow_pipeline.training_data[i]
            if label == 0:
                actual[3] = 5
            else:
                actual[3] = 6
            bow_pipeline.training_data[i] = (label, actual)

    @staticmethod
    def override_3th_index_in_attention_dataset(attention_pipeline: AttentionPipeline):
        for i in range(len(attention_pipeline.training_data)):
            label, actual, offset_begin, offset_end = attention_pipeline.training_data[i]
            if label == PipelineUtils.label_to_idx('FAVOR'):
                actual[3] = 5
            else:
                actual[3] = 6
            attention_pipeline.training_data[i] = (label, actual, offset_begin, offset_end)

    @staticmethod
    def set_only_tenth_of_dataset(model_pipeline: ModelPipeline):
        model_pipeline.training_data = model_pipeline.training_data[:(len(model_pipeline.training_data) // 10)]
        model_pipeline.validation_data = model_pipeline.validation_data[:(len(model_pipeline.validation_data) // 10)]
