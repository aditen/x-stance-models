import time
from typing import Dict, Tuple

import torch

from models.models.bow.bag_of_ngrams import BagOfNGrams
from models.pipelines.evaluation.model_evaluation_pipeline import ModelEvaluationPipeline
from models.pipelines.pipeline_utils import PipelineUtils
from models.properties.bow_properties import BoWModelProperties


class BoWEvaluationPipeline(ModelEvaluationPipeline):
    def __init__(self, model_props: BoWModelProperties):
        ModelEvaluationPipeline.__init__(self, model_props=model_props)
        self.model = BagOfNGrams(model_props.embedding_dim, self.embedding_pipeline.embedding_bag_layer,
                                 model_props.embedding is None).cpu().eval()
        self.model.load_state_dict(torch.load(".models/model_" + self.model_props.model_name))

    def create_input_tensors(self, question: str, comment: str) -> Tuple:
        tokens = self.embedding_pipeline.tokenize(question=question, answer=comment)
        return torch.tensor(tokens).cpu(), torch.tensor([0]).cpu()

    def evaluate(self, question: str, comment: str) -> Dict:
        start = time.time()
        tokens, offsets = self.create_input_tensors(question=question, comment=comment)
        with torch.no_grad():
            result = self.model(tokens, offsets)
            return {"tokens": [self.embedding_pipeline.itos[x] for x in
                               self.embedding_pipeline.tokenize(question, comment)],
                    "result": PipelineUtils.idx_to_label(result.round().item()),
                    "modelEvaluationDuration": time.time() - start}

    def write_weight_tsv(self):
        weight_matrix = self.model.embedding.weight.data[4:, :]
        with open(".results/" + self.model_props.model_name + "_embedding_projector.tsv", "w",
                  encoding="utf8") as file:
            # iterate over tokens
            for idx in range(len(weight_matrix)):
                # iterate over all weights but the last
                for idx_per_token in range(len(weight_matrix[idx]) - 1):
                    file.write(str(weight_matrix[idx, idx_per_token].item()) + "\t")
                file.write(str(weight_matrix[idx, len(weight_matrix[idx]) - 1].item()) + "\n")

        with open(".results/" + self.model_props.model_name + "_embedding_projector_labels.tsv", "w",
                  encoding="utf8") as file_labels:
            # single column should not have a header..
            # file_labels.write("Text\n")
            for val in self.embedding_pipeline.itos[4:]:
                file_labels.write(val + "\n")
