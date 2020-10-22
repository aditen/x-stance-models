import os

import simplejson as json
from sklearn.metrics import f1_score
from tqdm import tqdm

from models.pipelines.pipeline_utils import PipelineUtils

lang = "all"
if __name__ == "__main__":
    os.chdir("../..")
    outputs = []
    test_data = []
    test_actual_labels = []
    mbert_labels = []
    with open('.xstance/test.jsonl', encoding="utf-8") as f:
        for ind, objLine in enumerate(tqdm(f, unit="lines", desc="Test set lines", total=17705)):
            json_dict = json.loads(objLine)
            test_data.append(json_dict)
            test_actual_labels.append(PipelineUtils.label_to_idx(json_dict['label']))

    with open('.xstance/mbert_pred.jsonl', encoding="utf-8") as f:
        for ind, objLine in enumerate(tqdm(f, unit="lines", desc="MBert lines", total=17705)):
            mbert_labels.append(PipelineUtils.label_to_idx(json.loads(objLine)['label']))

    test_actual = []
    mbert_actual = []
    for i in range(len(test_actual_labels)):
        if lang == "all" or test_data[i]['language'] == lang:
            test_actual.append(test_actual_labels[i])
            mbert_actual.append(mbert_labels[i])

    print(f'F1 Score of lang {lang} is: {f1_score(test_actual, mbert_actual, average="macro") * 100:.2f}%')
