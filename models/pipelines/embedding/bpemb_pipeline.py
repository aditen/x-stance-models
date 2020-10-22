import jsonlines
import torch
from bpemb import BPEmb
from torch.nn import EmbeddingBag, Embedding
from torchtext.data.utils import ngrams_iterator
from tqdm import tqdm

from models.pipelines.embedding.embedding_pipeline import EmbeddingPipeline


class BpembPipeline(EmbeddingPipeline):

    def __init__(self, bp_emb_size, n_grams=5, freeze=True):
        EmbeddingPipeline.__init__(self, n_grams=n_grams, emb_id="bpemb_" + str(bp_emb_size), freeze=freeze)
        self.bp_emb_size = bp_emb_size
        self.bpemb_inst = BPEmb(lang="multi", vs=bp_emb_size, dim=300)
        self.__generate_embedding_and_dicts()

    def __generate_embedding_and_dicts(self):
        if not self.exists_on_disk():
            print("Preparing vocabulary, this may take a while. Please stay patient!")

            tok_idx = 0
            for i in range(len(self.bpemb_inst.pieces)):
                self.stoi[self.bpemb_inst.pieces[i]] = tok_idx
                self.itos.append(self.bpemb_inst.pieces[i])
                self.vecs.append(self.bpemb_inst.vectors[i])
                tok_idx += 1

            # add SEP, PAD and CLS tokens. Separator token needs to be added by index. Bpemb already contains a <unk> token!
            self.stoi[self.get_sep_string()] = tok_idx
            self.itos.append(self.get_sep_string())
            self.vecs.append([0.] * 300)
            tok_idx += 1

            self.stoi[self.get_pad_string()] = tok_idx
            self.itos.append(self.get_pad_string())
            self.vecs.append([0.] * 300)
            tok_idx += 1

            self.stoi[self.get_class_string()] = tok_idx
            self.itos.append(self.get_class_string())
            self.vecs.append((2 * torch.rand(300) - 1).tolist())
            tok_idx += 1

            single_vecs = list(self.vecs)

            line_count = 0

            with jsonlines.open(".xstance/train.jsonl") as lines:
                for line in tqdm(lines, unit='lines', total=45640):

                    encoded_ids = list(self.bpemb_inst.encode_ids(line['question'])) + [
                        self.stoi[self.get_sep_string()]] + list(self.bpemb_inst.encode_ids(line['comment']))

                    ngrams = list(ngrams_iterator(
                        [(self.bpemb_inst.pieces[int(i)] if int(i) < len(self.bpemb_inst.pieces) else self.itos[int(i)])
                         for i in encoded_ids], self.n_grams))

                    for tok in ngrams:
                        if tok not in self.stoi:
                            self.stoi[tok] = tok_idx
                            self.itos.append(tok)
                            vals = tok.split(' ')
                            aggregated = torch.stack(
                                [torch.tensor(single_vecs[self.stoi[val]]) for val
                                 in vals])
                            self.vecs.append(torch.mean(aggregated, dim=0).tolist())
                            tok_idx = tok_idx + 1

                    line_count += 1

            self.save_to_disk()
            self.embedding_bag_layer = EmbeddingBag.from_pretrained(torch.tensor(self.vecs), freeze=self.freeze)
            self.embedding_layer = Embedding.from_pretrained(torch.tensor(self.vecs), freeze=self.freeze)
        else:
            print("Loading merged embedding from disk. Should be fast!")
            self.load_from_disk()
            self.embedding_bag_layer = EmbeddingBag.from_pretrained(torch.tensor(self.vecs), freeze=self.freeze)
            self.embedding_layer = Embedding.from_pretrained(torch.tensor(self.vecs), freeze=self.freeze)

    def tokenize(self, question, answer, pad_to_length=None, include_class_token=False):
        encoded_ids = list(self.bpemb_inst.encode_ids(question)) + [
            self.stoi[self.get_sep_string()]] + list(self.bpemb_inst.encode_ids(answer))

        n_grams = list(ngrams_iterator(
            [(self.bpemb_inst.pieces[int(i)] if int(i) < len(self.bpemb_inst.pieces) else self.itos[int(i)]) for i in
             encoded_ids], self.n_grams))
        tkns = [(self.stoi[elem] if elem in self.stoi else self.stoi[self.get_unk_string()]) for elem in n_grams]

        return self._handle_padding_and_class_token(tkns, include_class_token=include_class_token,
                                                    pad_to_length=pad_to_length)
