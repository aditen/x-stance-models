import time
from typing import Tuple, Dict

import torch

from models.models.self_attention.custom_self_attention_model import CustomAttentionModel
from models.pipelines.evaluation.model_evaluation_pipeline import ModelEvaluationPipeline
from models.pipelines.pipeline_utils import PipelineUtils
from models.properties.attention_model_properties import AttentionModelProperties


class AttentionEvaluationPipeline(ModelEvaluationPipeline):
    def __init__(self, model_props: AttentionModelProperties):
        ModelEvaluationPipeline.__init__(self, model_props=model_props)
        self.model = CustomAttentionModel(vocab_size=len(self.embedding_pipeline.stoi),
                                          hidden_size=model_props.hidden_size, n_layers=model_props.num_layers,
                                          n_head=model_props.num_heads,
                                          dim_feedforward=model_props.intermediate_size,
                                          activation_function=model_props.activation_function,
                                          dropout=model_props.dropout,
                                          masking_strategies=model_props.masking_strategies).cpu().eval()
        self.model.load_state_dict(torch.load(".models/model_" + self.model_props.model_name))

    def create_input_tensors(self, question: str, comment: str) -> Tuple:
        tokens = self.embedding_pipeline.tokenize(question=question, answer=comment, pad_to_length=128,
                                                  include_class_token=True)
        sep_index = self.embedding_pipeline.stoi[self.embedding_pipeline.get_sep_string()]
        pad_index = self.embedding_pipeline.stoi[self.embedding_pipeline.get_pad_string()]
        end = 128
        if pad_index in tokens:
            end = tokens.index(pad_index)
        sequences = torch.zeros((128, 1), dtype=torch.long)
        print(torch.tensor(tokens).size())
        sequences[:, 0] = torch.tensor(tokens)
        return sequences.cpu(), torch.tensor([tokens.index(sep_index) + 1]).cpu(), torch.tensor([end]).cpu()

    def evaluate(self, question: str, comment: str) -> Dict:
        start = time.time()
        tokens, offset_begin, offset_end = self.create_input_tensors(question=question, comment=comment)
        with torch.no_grad():
            result, attn_weights = self.model(tokens, offset_begin, offset_end)
            return {"tokens": [self.embedding_pipeline.itos[x] for x in
                               self.embedding_pipeline.tokenize(question, comment, 128, True)],
                    "result": PipelineUtils.idx_to_label(result.round().item()),
                    "attnWeights": attn_weights[:, 0, :, :].tolist(),
                    "modelEvaluationDuration": time.time() - start}
