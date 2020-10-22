import jsonlines
import simplejson as json
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import TextClassificationDataset
from tqdm import tqdm

from models.models.self_attention.custom_self_attention_model import CustomAttentionModel
from models.pipelines.model.model_pipeline import ModelPipeline
from models.pipelines.pipeline_utils import PipelineUtils
from models.properties.attention_model_properties import AttentionModelProperties


class AttentionPipeline(ModelPipeline):
    def __init__(self, props: AttentionModelProperties):
        ModelPipeline.__init__(self, props, n_grams=1)
        self.generate_data_sets()
        self.model = CustomAttentionModel(vocab_size=len(self.embedding_pipeline.stoi), hidden_size=props.hidden_size,
                                          n_layers=props.num_layers, n_head=props.num_heads,
                                          dim_feedforward=props.intermediate_size,
                                          activation_function=props.activation_function, dropout=props.dropout,
                                          masking_strategies=props.masking_strategies).to(
            self.device)
        self.loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=props.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, props.step_size, gamma=props.gamma)
        self.batch_size = props.batch_size
        print("Initialized model")

    def generate_batch(self, batch):
        sequences = torch.zeros((128, len(batch)), dtype=torch.long)
        for i in range(len(batch)):
            sequences[:, i] = torch.tensor(batch[i][1])
        labels = torch.tensor([entry[0] for entry in batch], dtype=torch.float)
        begin_comments = torch.tensor([entry[2] for entry in batch])
        end_comments = torch.tensor([entry[3] for entry in batch])
        return labels.to(self.device), sequences.to(self.device), begin_comments.to(self.device), \
               end_comments.to(self.device)

    def generate_data_sets(self):
        with jsonlines.open('.xstance/train.jsonl') as reader:
            for obj in reader:
                self.training_data.append(self.__get_data_entry(obj))
            print("generated train dataset")

        with jsonlines.open('.xstance/valid.jsonl') as reader:
            for obj in reader:
                self.validation_data.append(self.__get_data_entry(obj))
            print("generated validation dataset")

    def train_single_epoch(self) -> (float, float):
        train_loss = 0
        train_acc = 0

        self.model.train()
        training_data = DataLoader(
            TextClassificationDataset(self.embedding_pipeline.stoi, self.training_data, PipelineUtils.get_labels()),
            batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)

        with tqdm(desc="Training progress", total=len(self.training_data), unit="samples", leave=False) as progress:
            for i, (labels, sequences, begin_comments, end_comments) in enumerate(training_data):
                progress.update(self.batch_size)
                self.optimizer.zero_grad()

                output, _ = self.model(sequences, begin_comments, end_comments)
                loss = self.loss(output[:, 0], labels)

                train_loss += loss.item()
                loss.backward()
                train_acc += (output[:, 0].round() == labels).sum().item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        # Adjust the learning rate
        self.scheduler.step()

        self.train_history.append((train_loss / len(self.training_data), train_acc / len(self.training_data)))

        return train_loss / len(self.training_data), train_acc / len(self.training_data)

    def validate_single_epoch(self) -> (float, float):
        self.model.eval()
        validation_loss = 0
        validation_accuracy = 0
        validation_data = DataLoader(
            TextClassificationDataset(self.embedding_pipeline.stoi, self.validation_data, PipelineUtils.get_labels()),
            batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        with tqdm(desc="Validation progress", total=len(self.validation_data), unit="samples", leave=False) as progress:
            for i, (labels, sequences, begin_comments, end_comments) in enumerate(validation_data):
                progress.update(self.batch_size)
                with torch.no_grad():
                    output, _ = self.model(sequences, begin_comments, end_comments)
                    loss = self.loss(output[:, 0], labels)
                    validation_loss += loss.item()
                    validation_accuracy += (output[:, 0].round() == labels).sum().item()

        validation_loss /= len(self.validation_data)
        validation_accuracy /= len(self.validation_data)

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.model = self.model.to("cpu")
            torch.save(self.model.state_dict(), self.model_path)
            self.model = self.model.to(self.device)

        self.validation_history.append((validation_loss, validation_accuracy))

        return validation_loss, validation_accuracy

    def evaluate_for_gui(self):
        self.model = self.model.to("cpu")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        with open('.xstance/test.jsonl', encoding="utf-8") as f:
            with torch.no_grad():
                for ind, objLine in enumerate(
                        tqdm(f, unit="lines", desc="Test set progress", total=17705, leave=False)):
                    obj = json.loads(objLine)
                    text = PipelineUtils.encode_with_separator(obj['question'], obj['comment'])
                    obj['encoded'] = text
                    obj['tokens'] = [self.embedding_pipeline.itos[i] for i in
                                     self.embedding_pipeline.tokenize(question=obj['question'], answer=obj['comment'],
                                                                      pad_to_length=128, include_class_token=True)]
                    obj['testSet'] = obj['test_set']
                    label, seq, offset_begin, offset_end = self.__get_data_entry(obj)
                    sequences = torch.zeros((128, 1), dtype=torch.long)
                    sequences[:, 0] = torch.tensor(seq)
                    output, attn_weights = self.model(
                        sequences,
                        torch.tensor([offset_begin]),
                        torch.tensor([offset_end])
                    )
                    obj['predicted'] = PipelineUtils.idx_to_label(
                        output.round().item()
                    )
                    self.results.append(obj)

        self.save_results_to_disk()

    def __get_data_entry(self, line):
        sep_index = self.embedding_pipeline.stoi[self.embedding_pipeline.get_sep_string()]
        pad_index = self.embedding_pipeline.stoi[self.embedding_pipeline.get_pad_string()]
        label = PipelineUtils.label_to_idx(line['label'])
        seq = self.embedding_pipeline.tokenize(line['question'], line['comment'],
                                               include_class_token=True, pad_to_length=128)
        end = 128
        if pad_index in seq:
            end = seq.index(pad_index)
        return label, seq, seq.index(sep_index) + 1, end
