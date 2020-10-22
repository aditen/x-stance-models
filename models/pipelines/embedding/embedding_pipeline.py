import os

import torch

"""
Interface for embeddings (vocab, tokenization, embedding layer)
"""


class EmbeddingPipeline:

    def __init__(self, n_grams=None, emb_id=None, min_token_occ=3, freeze=False):
        print("Initializing embedding")
        if emb_id is None or n_grams is None:
            raise ValueError("Embedding Pipeline ID and N Grams must be defined!")
        self.freeze = freeze
        self.emb_id = emb_id
        self.n_grams = n_grams
        self.min_token_occ = min_token_occ
        self.itos = []
        self.stoi = {}
        self.vecs = []
        self.embedding_path = '.embeddings/' + self.emb_id

    @staticmethod
    def get_unk_string() -> str:
        return "<unk>"

    @staticmethod
    def get_pad_string() -> str:
        return "<pad>"

    @staticmethod
    def get_sep_string() -> str:
        return "<sep>"

    @staticmethod
    def get_class_string() -> str:
        return "<cls>"

    """
    Embedding Layer, stoi dict und itos dict generation (or loading if already generated)
    """

    def __generate_embedding_and_dicts(self):
        pass

    """
    Tokenize a given question and answer
    """

    def tokenize(self, question, answer, pad_to_length=None, include_class_token=False):
        pass

    """
    Preprend CLS token if necessary and pad to length
    """

    def _handle_padding_and_class_token(self, tkns, include_class_token, pad_to_length):
        if include_class_token:
            tkns = [self.stoi[self.get_class_string()]] + tkns

        if pad_to_length is not None:
            if len(tkns) > pad_to_length:
                tkns = tkns[:pad_to_length]
            else:
                tkns = tkns + ([self.stoi[self.get_pad_string()]] * (pad_to_length - len(tkns)))
        return tkns

    def save_to_disk(self):
        torch.save({'itos': self.itos, 'stoi': self.stoi, 'vecs': self.vecs}, self.embedding_path)

    def load_from_disk(self):
        res = torch.load(self.embedding_path)
        self.itos, self.stoi, self.vecs = res['itos'], res['stoi'], res['vecs']

    def exists_on_disk(self):
        return os.path.isfile(self.embedding_path)
