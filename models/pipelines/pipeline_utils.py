import simplejson as json
from torchtext.data import get_tokenizer
from torchtext.data.utils import ngrams_iterator

LABELS_TO_IDX = {'FAVOR': 1, 'AGAINST': 0}
IDX_TO_LABEL = {0: 'AGAINST', 1: 'FAVOR'}

word_tokenizer_fc = get_tokenizer("basic_english")


class PipelineUtils:
    @staticmethod
    def get_labels():
        return set(LABELS_TO_IDX.values())

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

    @staticmethod
    def tokenize_words_using_vocab(vocab, sequence, n_grams):
        return [vocab[i] for i in
                ngrams_iterator(word_tokenizer_fc(sequence), n_grams)]

    @staticmethod
    def tokenize_words_using_dict(d, sequence, n_grams):
        return [(d[i] if i in d else 0) for i in
                ngrams_iterator(word_tokenizer_fc(sequence), n_grams)]

    @staticmethod
    def build_n_grams(tokens, n_grams):
        return ngrams_iterator(tokens, n_grams)

    @staticmethod
    def label_to_idx(label):
        return LABELS_TO_IDX[label]

    @staticmethod
    def idx_to_label(idx):
        return IDX_TO_LABEL[idx]

    @staticmethod
    def encode_with_separator(question, answer):
        return question + " " + PipelineUtils.get_sep_string() + " " + answer

    @staticmethod
    def encode_with_xlm_separator(question, answer):
        return question + " </s> " + answer

    @staticmethod
    def word_tokenize(text):
        return word_tokenizer_fc(text)

    @staticmethod
    def add_other_model_predictions(results):
        with open('.xstance/fasttext_pred.jsonl', encoding="utf-8") as f:
            for ind, objLine in enumerate(f):
                obj = json.loads(objLine)
                results[ind]['fasttext'] = obj['label']

        with open('.xstance/mbert_pred.jsonl', encoding="utf-8") as f:
            for ind, objLine in enumerate(f):
                obj = json.loads(objLine)
                results[ind]['mbert'] = obj['label']
