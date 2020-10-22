import os

from models.pipelines.evaluation.bow_evaluation_pipeline import BoWEvaluationPipeline
from models.properties.bow_properties import BoWModelProperties

if __name__ == "__main__":
    os.chdir("../..")
    BoWEvaluationPipeline(BoWModelProperties.tiny_own_embedding()).write_weight_tsv()
