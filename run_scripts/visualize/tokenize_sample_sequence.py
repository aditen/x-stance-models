import os

from models.debugging_utils import XStanceDebuggingUtils
from models.pipelines.embedding.own_word_embedding_pipeline import OwnWordEmbeddingPipeline

if __name__ == "__main__":
    os.chdir("../..")
    print("hello")
    embedding_pipeline = OwnWordEmbeddingPipeline(n_grams=1, embedding_dim=32)
    tkns = embedding_pipeline.tokenize("Ist er Politiker?", "Ja das ist er. Und auch ein Magier!")
    print(tkns)
    print("Actual tokens:")
    XStanceDebuggingUtils.print_tokens_based_on_embedding(tokens=tkns, embedding=embedding_pipeline)
