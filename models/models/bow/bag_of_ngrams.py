import torch
import torch.nn as nn


## see https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
class BagOfNGrams(nn.Module):
    def __init__(self, embed_dim, embedding_layer, zero_special_tokens=False):
        super(BagOfNGrams, self).__init__()
        self.embedding = embedding_layer
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim, 1)
        self.sig = nn.Sigmoid()
        self.init_weights(zero_special_tokens)

    def init_weights(self, zero_special_tokens):
        if zero_special_tokens:
            self.embedding.weight.data[0] = torch.zeros(self.embed_dim)
            self.embedding.weight.data[1] = torch.zeros(self.embed_dim)
            self.embedding.weight.data[2] = torch.zeros(self.embed_dim)
            self.embedding.weight.data[3] = torch.zeros(self.embed_dim)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.sig(self.fc(embedded))
