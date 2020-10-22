import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from models.pipelines.embedding.bpemb_pipeline import BpembPipeline
from models.pipelines.embedding.fasttext_pipeline import FasttextPipeline
from models.pipelines.embedding.own_word_embedding_pipeline import OwnWordEmbeddingPipeline
from models.properties.bow_properties import BoWModelProperties
from models.properties.model_properties import ModelProperties


class ModelEvaluationPipeline(ABC):
    def __init__(self, model_props: ModelProperties):
        if not os.path.isfile(".models/model_" + model_props.model_name):
            raise ValueError("Model " + model_props.model_name + " not existing, please train it first!")
        self.model_props = model_props
        if model_props.embedding == "fasttext":
            self.embedding_pipeline = FasttextPipeline(
                n_grams=model_props.n_grams if isinstance(model_props, BoWModelProperties) else 1)
        elif model_props.embedding == "bpemb":
            if model_props.bp_emb_size is None or model_props.bp_emb_size not in [100000, 320000, 1000000]:
                raise ValueError("BP Emb Size must be defined and in [100000, 320000, 1000000]!")
            self.embedding_pipeline = BpembPipeline(bp_emb_size=model_props.bp_emb_size,
                                                    n_grams=model_props.n_grams if isinstance(model_props,
                                                                                              BoWModelProperties) else 1)
        elif model_props.embedding is None:
            self.embedding_pipeline = OwnWordEmbeddingPipeline(
                n_grams=model_props.n_grams if isinstance(model_props, BoWModelProperties) else 1,
                embedding_dim=model_props.embedding_dim if isinstance(model_props,
                                                                      BoWModelProperties) else model_props.hidden_size)
        else:
            raise ValueError("Illegal value for embedding!")

    @abstractmethod
    def create_input_tensors(self, question: str, comment: str) -> Tuple:
        pass

    @abstractmethod
    def evaluate(self, question: str, comment: str) -> Dict:
        pass
