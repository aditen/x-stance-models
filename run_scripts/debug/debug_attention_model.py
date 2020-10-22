import os

from models.debugging_utils import XStanceDebuggingUtils
from models.pipelines.model.self_attention_pipeline import AttentionPipeline
from models.properties.attention_model_properties import AttentionModelProperties

if __name__ == "__main__":
    os.chdir("../..")

    pipeline = AttentionPipeline(AttentionModelProperties.small_ordinary())
    pipeline.model_name = "debugging_model"
    pipeline.model_path = ".models/debugging_model"

    XStanceDebuggingUtils.set_only_tenth_of_dataset(pipeline)
    XStanceDebuggingUtils.override_3th_index_in_attention_dataset(pipeline)

    pipeline.train_and_validate_for_epochs_and_test(evaluate_for_gui=False)
