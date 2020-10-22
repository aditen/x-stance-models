import torch
from torch.nn import Module, Embedding, Linear, Sigmoid

from models.models.self_attention.attention_weight_returning_transformer import TransformerEncoderWithAttention
from models.models.self_attention.positional_encoding import PositionalEncoding


class CustomAttentionModel(Module):
    def __init__(self, vocab_size, hidden_size=512, n_layers=2, n_head=2, dim_feedforward=1024,
                 activation_function="gelu", dropout=0., masking_strategies=None):
        super(CustomAttentionModel, self).__init__()
        if masking_strategies is None:
            masking_strategies = ["pad", "pad"]
        self.d_model = hidden_size
        self.embedding = Embedding(vocab_size, hidden_size)
        # zero out <unk> token and <pad> token
        self.embedding.weight.data[0] = torch.zeros(hidden_size)
        self.embedding.weight.data[1] = torch.zeros(hidden_size)
        self.positional_encoder = PositionalEncoding(hidden_size, dropout=dropout, max_len=128)
        self.encoder = TransformerEncoderWithAttention(n_layers=n_layers, d_model=hidden_size, nhead=n_head,
                                                       dim_feedforward=dim_feedforward, activation=activation_function,
                                                       dropout=dropout, masking_strategies=masking_strategies)

        self.fc = Linear(hidden_size, 1)
        self.sig = Sigmoid()

        # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

    def forward(self, tokens, offsets_comment_begin, offset_comment_end):
        embedded = self.embedding(tokens)
        positionally_encoded = self.positional_encoder(embedded)
        # mask should mask all padding indices
        # uses cls embedding
        output, attn_weights = self.encoder(positionally_encoded, offsets_comment_begin, offset_comment_end)

        output = self.sig(self.fc(output[0, :, :]))
        return output, attn_weights
