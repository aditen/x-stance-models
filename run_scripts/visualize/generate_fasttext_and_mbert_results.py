import os

import simplejson as json
from tqdm import tqdm

from models.pipelines.pipeline_utils import PipelineUtils

if __name__ == "__main__":
    os.chdir("../..")
    outputs = []
    with open('.xstance/test.jsonl', encoding="utf-8") as f:
        for ind, objLine in enumerate(tqdm(f, unit="lines", desc="Test set lines", total=17705)):
            obj = json.loads(objLine)
            text = PipelineUtils.encode_with_separator(obj['question'], obj['comment'])
            obj['encoded'] = text
            obj['tokens'] = PipelineUtils.word_tokenize(text)
            obj['testSet'] = obj['test_set']
            outputs.append(obj)

    with open('.xstance/mbert_pred.jsonl', encoding="utf-8") as f:
        for ind, objLine in enumerate(tqdm(f, unit="lines", desc="MBert prediction lines", total=17705)):
            obj = json.loads(objLine)
            outputs[ind]['predicted'] = obj['label']

    result_obj = {'predictions': outputs, 'modelId': "bert",
                  'trainingLossHistory': [],
                  'trainingAccuracyHistory': [],
                  'validationLossHistory': [],
                  'validationAccuracyHistory': []}

    with open(
            '.results/results_from_script_mbert.json', 'w',
            encoding="utf8") as fp:
        fp.flush()
        fp.write(str(json.dumps(result_obj, iterable_as_array=True, ensure_ascii=False)))

    with open('.xstance/fasttext_pred.jsonl', encoding="utf-8") as f:
        for ind, objLine in enumerate(tqdm(f, unit="lines", desc="Fasttext prediction lines", total=17705)):
            obj = json.loads(objLine)
            outputs[ind]['predicted'] = obj['label']

    result_obj = {'predictions': outputs, 'modelId': "fasttext_library",
                  'trainingLossHistory': [],
                  'trainingAccuracyHistory': [],
                  'validationLossHistory': [],
                  'validationAccuracyHistory': []}
    with open(
            '.results/results_from_script_fasttext_library.json', 'w',
            encoding="utf8") as fp:
        fp.flush()
        fp.write(str(json.dumps(result_obj, iterable_as_array=True, ensure_ascii=False)))
