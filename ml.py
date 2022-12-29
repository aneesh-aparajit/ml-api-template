import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import pandas as pd
import tez


class BERTDataset:
    def __init__(self, texts, targets, max_len=64) -> None:
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, ix):
        text = str(self.texts[ix])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_len=self.max_len,
            padding="max_len",
            truncation=True,
        )

        resp = {
            "ids": torch.tensor(inputs=["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[ix], dtype=torch.long),
        }

        return resp


class TextModel(tez.Model):
    def __init__(self, num_clases, num_train_steps):
        super().__init__()
        self.bert = (
            transformers.BertModel.from_pretrained(
                "bert-base-uncased", return_dict=False
            ),
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_clases)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        return {"accuracy": metrics.accuracy_score(targets, outputs)}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets)
            metric = self.monitor_metrics(x, targets)
            return x, loss, metric
        return x, 0, {}


def train_model(fold):
    df = pd.read_csv("./imdb_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold != fold].reset_index(drop=True)

    train_dataset = BERTDataset(df_train.review.values, df_train.sentiment.values)
    valid_dataset = BERTDataset(df_valid.reviews.values, df_valid.sentiment.values)

    n_train_steps = int(len(df_train) / 32 * 10)
    model = TextModel(num_clases=1, num_train_steps=n_train_steps)

    early_stopping = tez.callbacks.EarlyStopping(
        monitor="valid_loss", patience=3, model_path="model.bin"
    )

    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device="mps",
        epochs=10,
        train_bs=32,
        callbacks=[early_stopping],
    )
