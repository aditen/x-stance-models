# X Stance task

# Purpose
This is the repository containing all Python code related to my Bachelor's thesis. 
It basically deals with three problems:
- Creating a vocabulary and embedding layers for a custom or pre-trained embedding on the X-Stance task
- Building models for the XStance task: We evaluate a BoW as well as Self-Attention based models
- Evaluating the models's performance for linguistic error analysis

# Setup (with instructions for Windows environment)
- **Prerequisite**: Python 3.7+ **64-bit** installation 
- Installation:
    - Install PyTorch (see https://pytorch.org/get-started/locally/)
    - Create a virtual environment in the code directory, using command `python -m venv venv`
    - Install required packages using command `pip install -r requirements.txt`
    - run setup script to create folder and download data set using command `python setup.py`
    - Train just the tiny bow model using command `python train_tiny_model.py`
- **Automated installation** on windows, run the batch file `project_setup.bat`

# Testing setup
In the end of the automatic setup script, the script train_tiny_model.py is executed - this may also be done manually in order to test 
that torch and the whole project was set up correctly using the command ``python train_tiny_model.py`` - in case this runs through and 
writes the model's results to .results, the setup is fine


# Training and evaluating thesis models
Simply run train_thesis_models script using command `python train_thesis_models.py` or start the script in your IDE
At the beginning of the script, for every experiment section it is possible to choose whether to train those models or not.


# File structure
The files are structured as follows:
- models: Contains all model related classes, but no run scripts
    - models: Actual PyTorch model implementations (inheriting from nn.Module) 
    with the initialization of the model and it's forward path
        - bow
            - bag_of_ngrams.py: Bag of Words (respectively N-Grams if given) model
        - self_attention:
            - attention_weight_returning_transformer.py: Copy of PyTorch's transformer encoder and transformer encoder layer, returning the attention weights from the forward method too
            - custom_self_attention_model.py: A model stacking n Self-Attention based Encoder layers and then applying a linear classifier as described in the thesis
            - positional_encoding.py: Implementation of Positional Encoding as described in thesis
    - pipelines: Pipelines abstracting models and embeddings as uniform interfaces
        - embedding: Pipelines to create Embedding(-Bag) layers and token dictionaries 
            - embedding_pipeline.py: Base interface 
            - own_word_embedding_pipeline.py: Custom embeddings (vocab from train data only)
            - fasttext_pipeline.py: Fasttext (vocab from train, validate and test set! 
            merging all languages would be 3'000'000+ tokens, too big for a small model)
            - bpemb_pipeline.py: BPEmb (pre-defined vocab)
        - model: Pipelines for a model with functions to: train it, validate it and test it as well as 
        writing JSON for X-Stance Dashboard
            - model_pipeline.py: Abstract class of model pipeline: defining functions to train, validate and test models 
            as well as implementing function to save
            - self_attention_pipeline.py: Implementation of Pipeline for Self Attention Models, using custom_self_attention_model
            - bow_pipeline.py: Implementation of Pipeline for BoW model
        - properties: Data Class that holds properties of models including hyper parameters that define the model
            - model_properties.py: Base class that handles common attributes like name, epochs, batch size and others
            - attention_model_properties.py: Attention Model Properties, adding self-attention specific properties 
            like number of encoder layers, heads and more
            - bow_properties.py: BoW Model Properties, adding BoW specific properties like the NGram window
        - evaluation: Evaluation of Models, either actually evaluating them on any given input or 
        mapping parameters for external analysis like the embedding projector
            - model_evaluation_pipeline.py: Base Class that initializes some values and defines interface
            - bow_evaluation_pipeline.py: Implementation of Evaluation for BoW Model, also including function 
            to write embedding space to file for embedding projector
            - self_attention_evaluation_pipeline.py: Implementation of Evaluation for Attention Model, 
            also returning attention weights  
        - pipeline_utils: General Utilities that are used like Index <-> Label handling and Encoding of Sequence functions
    - debugging_utils.py: Function that may be used to debug a model's functionality or pipeline
- run_scripts: Contains specific python run scripts used during development
    - debug
        - debug_attention_model.py: Script to debug attention based models
        - debug_bp_emb_vocab.py: Just visualizing how bp emb actually tokenizes a sequence
    - visualize: Contains help scripts to get e. g. f1 score (used to validate X-Stance Dashboard's calculation), 
    write results for baseline models and some that are just a run script for a pipeline
- train_thesis_models.py: Trains all models whose results are in the thesis **ATTENTION: THIS TAKES LONG!**
- setup.py: Python script that downloads the dataset and creates directories used for embeddings and results
- web_app.py: Runs a web app on port 5000 that offers a REST API to evaluate selected models


# Evaluating models in X-Stance Dashboard or Tensorflow Projector
The model pipelines can be called with the method train_and_validate_for_epochs_and_test and by default also write
the JSON file used by the GUI to the .results directory. It may then be moved to wherever the file server of the GUI 
reads files from in order to display it in the X-Stance Dashboard

# Running evaluation with docker
First, build the image (replace eval-test with your desired name) using the following command: ``docker build -t eval-test .``
Then, run it with the following command: ``docker run -p 5000:5000 -v /var/you_models_location:/var/model_app/.models eval-test``
Replace /var/you_models_location with the location where you store the models

The evaluation app is also deployed on https://x-stance-eval.iten.engineering

It has two endpoints: /predict_bow and /predict_attention
Both expect a JSON body consistng of a question and comment attribute, for example:

``
{"question": "Ist das wichtig?", "comment": "Ja wichtig, sehr wichtig sogar. Bürokratie ftw"}
``

It then again returns a status code of 200 if a prediction can be made and a JSON of the following shape. The attention pipeline
contains further a 3D double matrix as ``attnWeights`` property:

``
{
    "tokens": [
        "ist",
        "das",
        "wichtig",
        "?",
        "<sep>",
        "ja",
        "wichtig",
        ",",
        "sehr",
        "wichtig",
        "sogar",
        ".",
        "bürokratie",
        "<unk>"
    ],
    "result": "FAVOR",
    "modelEvaluationDuration": 0.0024328231811523438
}
``

# ToDos:
- Use XXModelProperties in Actual models to determine the parameters instead of passing them from pipeline 
