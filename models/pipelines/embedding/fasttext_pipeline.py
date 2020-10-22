import jsonlines
import torch
from torch.nn import EmbeddingBag, Embedding
from torchnlp.word_to_vector.fast_text import FastText
from tqdm import tqdm

from models.pipelines.embedding.embedding_pipeline import EmbeddingPipeline
from models.pipelines.pipeline_utils import PipelineUtils


class FasttextPipeline(EmbeddingPipeline):
    def __init__(self, n_grams=5, min_token_occ=1, freeze=True):
        EmbeddingPipeline.__init__(self, n_grams=n_grams, emb_id="fasttext", min_token_occ=min_token_occ, freeze=freeze)
        self.it: FastText = FastText(language="it", aligned=True)
        self.de: FastText = FastText(language="de", aligned=True)
        self.fr: FastText = FastText(language="fr", aligned=True)
        self.__generate_embedding_and_dicts()
        print("Initialized embedding")

    def __generate_embedding_and_dicts(self):
        if not self.exists_on_disk():
            print("Preparing vocabulary, this may take a while. Please stay patient!")

            # add UNK, PAD, SEP and CLS tokens
            self.stoi[self.get_unk_string()] = 0
            self.itos.append(self.get_unk_string())
            self.vecs.append([0.] * 300)
            self.stoi[self.get_pad_string()] = 1
            self.itos.append(self.get_pad_string())
            self.vecs.append([0.] * 300)
            self.stoi[self.get_class_string()] = 2
            self.itos.append(self.get_class_string())
            self.vecs.append((2 * torch.rand(300) - 1).tolist())
            self.stoi[self.get_sep_string()] = 3
            self.itos.append(self.get_sep_string())
            self.vecs.append([0.] * 300)

            tok_idx = 4
            occ_dict = {}
            with jsonlines.open(".xstance/all.jsonl") as lines:
                for line in tqdm(lines, unit='lines', total=67271):
                    n_grams = PipelineUtils.build_n_grams(
                        PipelineUtils.word_tokenize(
                            PipelineUtils.encode_with_separator(line['question'], line['comment'])),
                        self.n_grams)

                    lin_lang = line['language']
                    for tok in n_grams:
                        if tok not in occ_dict:
                            occ_dict[tok] = 1
                        else:
                            occ_dict[tok] += 1

            for key in occ_dict:
                if occ_dict[key] >= self.min_token_occ:
                    self.stoi[key] = tok_idx
                    self.itos.append(key)
                    vals = key.split(' ')
                    if lin_lang == 'fr':
                        vec = torch.stack([(self.fr[x] if x in self.fr else torch.zeros(300)) for x in vals])
                    elif lin_lang == 'de':
                        vec = torch.stack([(self.de[x] if x in self.de else torch.zeros(300)) for x in vals])
                    elif lin_lang == 'it':
                        vec = torch.stack([(self.it[x] if x in self.it else torch.zeros(300)) for x in vals])
                    else:
                        raise ValueError("Found line that is neither de, nor fr or it")

                    vec = vec.mean(dim=0).tolist()
                    self.vecs.append(vec)
                    tok_idx += 1

            self.save_to_disk()
            self.embedding_bag_layer, self.embedding_layer = EmbeddingBag.from_pretrained(
                torch.tensor(self.vecs), freeze=self.freeze), Embedding.from_pretrained(
                torch.tensor(self.vecs), freeze=self.freeze
            )

        else:
            print("Loading merged embedding from disk. Should be fast!")
            self.load_from_disk()
            self.embedding_bag_layer, self.embedding_layer = EmbeddingBag.from_pretrained(
                torch.tensor(self.vecs), freeze=self.freeze), Embedding.from_pretrained(
                torch.tensor(self.vecs), freeze=self.freeze
            )

    def tokenize(self, question, answer, pad_to_length=None, include_class_token=False):
        tkns = [(self.stoi[tok] if tok in self.stoi else self.stoi[self.get_unk_string()]) for tok in
                PipelineUtils.build_n_grams(
                    PipelineUtils.word_tokenize(PipelineUtils.encode_with_separator(question, answer)),
                    self.n_grams)]
        return self._handle_padding_and_class_token(tkns, include_class_token=include_class_token,
                                                    pad_to_length=pad_to_length)

    def load_vecs(self):
        return torch.load(".vecs/vecs_fasttext")
