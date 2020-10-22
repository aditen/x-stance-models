import copy
from typing import Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, functional as F, Module, ModuleList


class TransformerEncoderWithAttention(Module):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 masking_strategies=None):
        super(TransformerEncoderWithAttention, self).__init__()
        if masking_strategies is None or len(masking_strategies) != n_layers:
            masking_strategies = ["pad"] * n_layers
            print("set masking to default strategy: pad")
        if masking_strategies[-1] != "pad":
            raise ValueError("Masking strategies needs to end with a pad strategy!")
        self.masking_strategies = masking_strategies
        self.encoding_Layers = _get_clones(
            TransformerEncoderLayerWithAttention(d_model, nhead, dim_feedforward, dropout, activation), n_layers)

    def forward(self, src: Tensor, begin_comment_offsets, end_comment_offsets) -> Tuple[Tensor, List[Tensor]]:
        attn_list = []
        for i, layer in enumerate(self.encoding_Layers):
            src, attn_weights = layer(src,
                                      src_key_padding_mask=TransformerEncoderWithAttention.__generate_only_look_at_question_mask(
                                          begin_comment_offsets, end_comment_offsets) if self.masking_strategies[
                                                                                             i] == "question_only" else (
                                          TransformerEncoderWithAttention.__generate_only_look_at_comment_mask(
                                              begin_comment_offsets,
                                              end_comment_offsets) if self.masking_strategies[
                                                                          i] == "comment_only" else TransformerEncoderWithAttention.__generate_ignore_padding_mask(
                                              begin_comment_offsets,
                                              end_comment_offsets)))
            attn_list.append(attn_weights)
        return src, torch.stack(attn_list)

    # custom function: generates attention mask on padding indices!
    # mask dimensions: N(batch size) x S (sequence length)
    @staticmethod
    def __generate_ignore_padding_mask(offset_comment_begin, offsets_comment_end):
        mask = torch.zeros((len(offsets_comment_end), 128), dtype=torch.bool).to(offsets_comment_end.device)
        for i in range(0, len(offsets_comment_end)):
            val = offsets_comment_end[i].item()
            if val < 128:
                mask[i][val:] = True
            else:
                pass
        return mask

    # custom function: generates attention mask on padding indices!
    # mask dimensions: N(batch size) x S (sequence length)
    @staticmethod
    def __generate_only_look_at_question_mask(offset_comment_begin, offsets_comment_end):
        mask = torch.ones((len(offsets_comment_end), 128), dtype=torch.bool).to(offsets_comment_end.device)
        for i in range(0, len(offsets_comment_end)):
            val = offset_comment_begin[i].item()
            mask[i][:val] = False
            mask[i][0] = True
        return mask

    # custom function: generates attention mask on padding indices!
    # mask dimensions: N(batch size) x S (sequence length)
    @staticmethod
    def __generate_only_look_at_comment_mask(offset_comment_begin, offsets_comment_end):
        mask = torch.ones((len(offsets_comment_end), 128), dtype=torch.bool).to(offsets_comment_end.device)
        for i in range(0, len(offsets_comment_end)):
            begin_idx = offset_comment_begin[i].item()
            end_idx = offsets_comment_end[i].item()
            mask[i][begin_idx:end_idx] = False
        return mask


class TransformerEncoderLayerWithAttention(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerWithAttention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            print("using RELU as not set in state!")
            state['activation'] = F.relu
        super(TransformerEncoderLayerWithAttention, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
