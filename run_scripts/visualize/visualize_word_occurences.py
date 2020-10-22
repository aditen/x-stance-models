import os

import matplotlib.pyplot as plt
import numpy as np
from jsonlines import jsonlines
from tqdm import tqdm

from models.pipelines.pipeline_utils import PipelineUtils

if __name__ == "__main__":
    os.chdir("../..")
    labels = [str(i + 1) for i in range(10)]
    labels[9] = "10+"

    x = np.arange(len(labels))  # the label locations
    width = 0.8 / len(labels)  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    data_labels = ["Unigrams", "Bigrams", "Trigrams", "Fourgrams", "Fivegrams"]

    for i in range(5):
        occ_dict = {}
        with jsonlines.open(".xstance/train.jsonl") as lines:
            for line in tqdm(lines, unit='lines', total=45640):
                n_grams = PipelineUtils.build_n_grams(
                    PipelineUtils.word_tokenize(
                        PipelineUtils.encode_with_separator(line['question'], line['comment'])),
                    i + 1)
                for tok in n_grams:
                    if tok not in occ_dict:
                        occ_dict[tok] = 1
                    else:
                        occ_dict[tok] += 1
            data = [0] * 10
            for key in occ_dict:
                if occ_dict[key] > 9:
                    data[9] += 1
                else:
                    data[occ_dict[key] - 1] += 1
            ax.bar(x - 2 * width + i * width, data, width, label=data_labels[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of tokens in vocabulary')
    ax.set_title('Number of tokens by their occurrence within the training set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xlabel("Number of times occurring in the training set")
    ax.legend()

    fig.tight_layout()

    plt.show()
