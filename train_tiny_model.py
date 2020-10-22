from models.pipelines.model.bow_pipeline import BowPipeline
from models.properties.bow_properties import BoWModelProperties

if __name__ == "__main__":
    print("Script started")
    BowPipeline(BoWModelProperties.tiny_own_embedding()).train_and_validate_for_epochs_and_test()
    print("Script terminated")
