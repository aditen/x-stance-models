from jsonlines import jsonlines
from torch.nn import EmbeddingBag, Embedding
from tqdm import tqdm

from models.pipelines.embedding.embedding_pipeline import EmbeddingPipeline
from models.pipelines.pipeline_utils import PipelineUtils


class OwnWordEmbeddingPipeline(EmbeddingPipeline):
    def __init__(self, n_grams=5, embedding_dim=300, min_token_occ=3):
        EmbeddingPipeline.__init__(self, n_grams=n_grams, emb_id="own_word_" + str(n_grams),
                                   min_token_occ=min_token_occ, freeze=False)
        self.embedding_dim = embedding_dim
        self.__generate_embedding_and_dicts()

    def __generate_embedding_and_dicts(self):
        if not self.exists_on_disk():
            print("Preparing vocabulary, this may take a while. Please stay patient!")
            # add UNK, PAD, SEP and CLS tokens
            self.stoi[self.get_unk_string()] = 0
            self.itos.append(self.get_unk_string())
            self.stoi[self.get_pad_string()] = 1
            self.itos.append(self.get_pad_string())
            self.stoi[self.get_class_string()] = 2
            self.itos.append(self.get_class_string())
            self.stoi[self.get_sep_string()] = 3
            self.itos.append(self.get_sep_string())

            tok_idx = 4

            occ_dict = {}
            with jsonlines.open(".xstance/train.jsonl") as lines:
                for line in tqdm(lines, unit='lines', total=45640):
                    n_grams = PipelineUtils.build_n_grams(
                        PipelineUtils.word_tokenize(
                            PipelineUtils.encode_with_separator(line['question'], line['comment'])),
                        self.n_grams)

                    for tok in n_grams:
                        if tok not in occ_dict:
                            occ_dict[tok] = 1
                        else:
                            occ_dict[tok] += 1

            for key in occ_dict:
                # and is necessary because of separator token..
                if occ_dict[key] >= self.min_token_occ and key not in self.stoi:
                    self.stoi[key] = tok_idx
                    self.itos.append(key)
                    tok_idx += 1

            self.save_to_disk()
        else:
            print("Loading merged embedding from disk. Should be fast!")
            self.load_from_disk()

        self.embedding_bag_layer = EmbeddingBag(len(self.stoi), self.embedding_dim)
        self.embedding_layer = Embedding(len(self.stoi), self.embedding_dim)

    def tokenize(self, question, answer, pad_to_length=None, include_class_token=False):
        tkns = [(self.stoi[tok] if tok in self.stoi else self.stoi[self.get_unk_string()]) for tok in
                PipelineUtils.build_n_grams(
                    PipelineUtils.word_tokenize(PipelineUtils.encode_with_separator(question, answer)),
                    self.n_grams)]
        return self._handle_padding_and_class_token(tkns, include_class_token=include_class_token,
                                                    pad_to_length=pad_to_length)
