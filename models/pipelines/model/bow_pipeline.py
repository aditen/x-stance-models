import jsonlines
import simplejson as json
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import TextClassificationDataset
from tqdm import tqdm

from models.models.bow.bag_of_ngrams import BagOfNGrams
from models.pipelines.model.model_pipeline import ModelPipeline
from models.pipelines.pipeline_utils import PipelineUtils
from models.properties.bow_properties import BoWModelProperties

allowed_embeddings = [None, "fasttext", "bpemb"]


class BowPipeline(ModelPipeline):

    def __init__(self, props: BoWModelProperties):
        ModelPipeline.__init__(self, props, n_grams=props.n_grams,
                               embedding_dim=props.embedding_dim,
                               bp_emb_size=props.bp_emb_size)
        self.generate_data_sets()
        self.model = BagOfNGrams(props.embedding_dim, self.embedding_pipeline.embedding_bag_layer,
                                 props.embedding is None).to(self.device)
        self.loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=props.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, props.step_size, gamma=props.gamma)
        self.batch_size = props.batch_size
        print("Initialized model")

    def generate_batch(self, batch):
        label = torch.tensor([entry[0] for entry in batch], dtype=torch.float)
        text = [torch.tensor(entry[1]) for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        # torch.Tensor.cumsum returns the cumulative sum
        # of elements in the dimension dim.
        # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text.to(self.device), offsets.to(self.device), label.to(self.device)

    def generate_data_sets(self):
        with jsonlines.open('.xstance/train.jsonl') as reader:
            for obj in reader:
                self.training_data.append(
                    (PipelineUtils.label_to_idx(obj['label']),
                     self.embedding_pipeline.tokenize(obj['question'], obj['comment']))
                )
            print("generated train dataset")

        with jsonlines.open('.xstance/valid.jsonl') as reader:
            for obj in reader:
                self.validation_data.append(
                    (PipelineUtils.label_to_idx(obj['label']),
                     self.embedding_pipeline.tokenize(obj['question'], obj['comment']))
                )
            print("generated validation dataset")

    def train_single_epoch(self) -> (float, float):
        train_loss = 0
        train_acc = 0
        training_data = DataLoader(
            TextClassificationDataset(self.embedding_pipeline.stoi, self.training_data, PipelineUtils.get_labels()),
            batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)

        with tqdm(desc="Training progress", total=len(self.training_data), unit="samples", leave=False) as progress:
            for i, (text, offsets, cls) in enumerate(training_data):
                progress.update(self.batch_size)
                self.optimizer.zero_grad()
                output = self.model(text, offsets)
                loss = self.loss(output[:, 0], cls)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                train_acc += (output[:, 0].round() == cls).sum().item()

        # Adjust the learning rate
        self.optimizer.step()

        train_loss /= len(self.training_data)
        train_acc /= len(self.training_data)

        self.train_history.append((train_loss, train_acc))

        return train_loss, train_acc

    def validate_single_epoch(self) -> (float, float):
        validation_loss = 0
        validation_acc = 0
        validation_data = DataLoader(
            TextClassificationDataset(self.embedding_pipeline.stoi, self.validation_data, PipelineUtils.get_labels()),
            batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)

        with tqdm(desc="Validation progress", total=len(self.validation_data), unit="samples", leave=False) as progress:
            for text, offsets, cls in validation_data:
                progress.update(self.batch_size)
                with torch.no_grad():
                    output = self.model(text, offsets)
                    loss = self.loss(output[:, 0], cls)
                    validation_loss += loss.item()
                    validation_acc += (output[:, 0].round() == cls).sum().item()

        validation_loss /= len(self.validation_data)
        validation_acc /= len(self.validation_data)

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.model = self.model.to("cpu")
            torch.save(self.model.state_dict(), self.model_path)
            self.model = self.model.to(self.device)

        self.validation_history.append((validation_loss, validation_acc))

        return validation_loss, validation_acc

    def evaluate_for_gui(self):
        self.model = self.model.to("cpu")
        self.model.load_state_dict(torch.load(self.model_path))

        print("generating results")

        with open('.xstance/test.jsonl', encoding="utf-8") as f:
            for ind, objLine in enumerate(tqdm(f, unit="lines", desc="Test set progress", total=17705, leave=False)):
                # if ind in test_indices:
                obj = json.loads(objLine)
                text = PipelineUtils.encode_with_separator(obj['question'], obj['comment'])
                obj['encoded'] = text
                obj['tokens'] = [self.embedding_pipeline.itos[i] for i in
                                 self.embedding_pipeline.tokenize(question=obj['question'], answer=obj['comment'])]
                obj['testSet'] = obj['test_set']
                model_input = torch.tensor(self.embedding_pipeline.tokenize(obj['question'], obj['comment']))
                with torch.no_grad():
                    output = self.model(model_input, torch.tensor([0]))
                    prediction = output[0, 0].round().item()
                    obj['predicted'] = PipelineUtils.idx_to_label(prediction)
                self.results.append(obj)

        self.save_results_to_disk()
