import time
from abc import ABC, abstractmethod

import numpy as np
import simplejson as json
import torch

from models.pipelines.embedding.bpemb_pipeline import BpembPipeline
from models.pipelines.embedding.fasttext_pipeline import FasttextPipeline
from models.pipelines.embedding.own_word_embedding_pipeline import OwnWordEmbeddingPipeline
from models.properties.model_properties import ModelProperties


class ModelPipeline(ABC):

    def __init__(self, model_props: ModelProperties, n_grams: int = 1, embedding_dim=300, bp_emb_size=None):
        print("Initializing model")
        if model_props.model_name is None:
            raise ValueError("Model Name and embedding must be provided!")
        self.model_props = model_props
        if model_props.embedding == "fasttext":
            self.embedding_pipeline = FasttextPipeline(n_grams=n_grams)
        elif model_props.embedding == "bpemb":
            if bp_emb_size is None or bp_emb_size not in [100000, 320000, 1000000]:
                raise ValueError("BP Emb Size must be defined and in [100000, 320000, 1000000]!")
            self.embedding_pipeline = BpembPipeline(bp_emb_size=bp_emb_size, n_grams=n_grams)
        elif model_props.embedding is None:
            self.embedding_pipeline = OwnWordEmbeddingPipeline(n_grams=n_grams, embedding_dim=embedding_dim)
        else:
            raise ValueError("Illegal value for embedding!")
        print("Vocab size:", len(self.embedding_pipeline.stoi))
        self.model_path = ".models/model_" + model_props.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_history = []
        self.validation_history = []
        self.results = []
        self.training_data = []
        self.validation_data = []
        self.total_training_time = -1
        self.min_validation_loss = float('inf')

    @abstractmethod
    def generate_batch(self, batch):
        pass

    @abstractmethod
    def generate_data_sets(self):
        pass

    @abstractmethod
    def train_single_epoch(self) -> (float, float):
        pass

    @abstractmethod
    def validate_single_epoch(self) -> (float, float):
        pass

    @abstractmethod
    def evaluate_for_gui(self):
        pass

    def train_and_validate_for_epochs_and_test(self, evaluate_for_gui=True):
        start_time = time.time()
        for epoch in range(self.model_props.n_epochs):
            # TRAINING
            train_loss, train_acc = self.train_single_epoch()
            # VALIDATING
            validation_loss, validation_accuracy = self.validate_single_epoch()
            print(
                f'Train loss: {train_loss:.4f} and Accuracy: {train_acc * 100:.1f}%, validation loss: '
                f'{validation_loss:.4f} and accuracy: {validation_accuracy * 100:.1f}% in epoch {epoch + 1}')
        self.total_training_time = (time.time() - start_time)

        print(
            f'Lowest validation loss: {self.min_validation_loss:.4f}, number of parameters: '
            f'{sum([np.prod(p.size()) for p in (filter(lambda p: p.requires_grad, self.model.parameters()))])}'
            f', total time: {self.total_training_time:.2f}s')

        if evaluate_for_gui:
            self.evaluate_for_gui()

    def save_results_to_disk(self):
        result_obj = {'predictions': self.results, 'modelId': self.model_props.model_name,
                      'trainingLossHistory': [elem[0] for elem in self.train_history],
                      'trainingAccuracyHistory': [elem[1] for elem in self.train_history],
                      'validationLossHistory': [elem[0] for elem in self.validation_history],
                      'validationAccuracyHistory': [elem[1] for elem in self.validation_history],
                      'trainingTime': self.total_training_time}

        result_path = '.results/results_from_script_' + self.model_props.model_name + '.json'
        with open(result_path, 'w', encoding="utf8") as fp:
            fp.flush()
            fp.write(str(json.dumps(result_obj, iterable_as_array=True, ensure_ascii=False)))

        print(f'written result to {result_path}')
