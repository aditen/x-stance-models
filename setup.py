import io
import os
import urllib.request
import zipfile

import requests

if __name__ == "__main__":

    if not os.path.exists(".xstance"):
        os.mkdir(".xstance")
        print("created .xstance data path")

    if not os.path.exists(".embeddings"):
        os.mkdir(".embeddings")
        print("created embeddings folder")

    if not os.path.exists(".models"):
        os.mkdir(".models")
        print("created models folder")

    if not os.path.exists(".results"):
        os.mkdir(".results")
        print("created results folder")

    if not os.path.exists(".xstance/fasttext_pred.jsonl"):
        # download fasttext baseline
        url = 'https://raw.githubusercontent.com/ZurichNLP/xstance/master/predictions/fasttext_pred.jsonl'
        response = urllib.request.urlopen(url)
        data = response.read()  # a `bytes` object
        text = data.decode('utf-8')[:-1]  # suppress last line break when converting
        text_file = open(".xstance/fasttext_pred.jsonl", "w")
        text_file.write(text)
        text_file.close()
        print("downloaded fasttext predictions")

    if not os.path.exists(".xstance/mbert_pred.jsonl"):
        # download bert baseline
        url = 'https://raw.githubusercontent.com/ZurichNLP/xstance/master/predictions/mbert_pred.jsonl'
        response = urllib.request.urlopen(url)
        data = response.read()  # a `bytes` object
        text = data.decode('utf-8')[:-1]  # suppress last line break when converting
        text_file = open(".xstance/mbert_pred.jsonl", "w")
        text_file.write(text)
        text_file.close()
        print("downloaded mbert predictions")

    # could also fetch those from huggingface transformers as it is included there I saw
    if not os.path.isfile(".xstance/train.jsonl") or not os.path.isfile(".xstance/valid.jsonl") or not os.path.isfile(
            ".xstance/test.jsonl") or not os.path.isfile(".xstance/all.jsonl"):
        # download and unpack zip with tests set, merge them to a "all" file
        url = "https://github.com/ZurichNLP/xstance/blob/master/data/xstance-data-v1.0.zip?raw=true"
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".xstance")

        with open(".xstance/test.jsonl", "r", encoding="utf8") as test, open(".xstance/valid.jsonl", "r",
                                                                             encoding="utf8") as valid, open(
            ".xstance/train.jsonl", "r", encoding="utf8") as train, open(".xstance/all.jsonl", "w",
                                                                         encoding="utf8") as all_file:
            all_file.write(train.read() + valid.read() + test.read())

        print("downloaded and extracted dataset")

    print("setup complete!")
