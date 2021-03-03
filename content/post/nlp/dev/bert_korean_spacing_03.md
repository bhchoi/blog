---
title: "BERT를 이용한 한국어 띄어쓰기 모델 만들기 - 03. 모델"
date: 2021-01-10 23:11:00 +0800
categories: [nlp]
tags: [bert, spacing, 띄어쓰기]
toc: true
---

## Config 설정

학습에 필요한 각종 값과 하이퍼파라미터입니다.  
argparse를 이용해 인자로 받을 수도 있으나, yaml로 정의하여 불러오도록 하겠습니다.

```yaml
task: korean_spacing_20210101
log_path: logs
bert_model: monologg/kobert
train_data_path: data/nikl_newspaper/train.txt
val_data_path: data/nikl_newspaper/val.txt
test_data_path: data/nikl_newspaper/test.txt
max_len: 128
train_batch_size: 128
eval_batch_size: 128
dropout_rate: 0.1
gpus: 8
distributed_backend: ddp
```
&nbsp;

## 모델 생성

<a href="https://github.com/PyTorchLightning/pytorch-lightning" target="_blank">PyTorch Lightning</a>을 이용해 모델을 생성해보겠습니다.  
PyTorch Lightning은 tensorflow의 keras처럼, 좀 더 쉽게 pytorch를 사용할 수 있는 라이브러리입니다.  


먼저, pytorch_lightning을 상속받아 클래스를 정의하고, 필요한 function들을 작성해줍니다.

```python
import pytorch_lightning as pl

class SpacingBertModel(pl.LightningModule):
    def __init__(self):
        pass
```
&nbsp;

dataset과 학습에 필요한 config를 넘겨받습니다.

```python
class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset: CorpusDataset,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
```
&nbsp;

dataloader를 생성합니다.

```python
def train_dataloader(self):
    return DataLoader(self.dataset["train"], batch_size=self.config.train_batch_size)

def val_dataloader(self):
    return DataLoader(self.dataset["val"], batch_size=self.config.eval_batch_size)

def test_dataloader(self):
    return DataLoader(self.dataset["test"], batch_size=self.config.eval_batch_size)
```
&nbsp;

모델을 정의하고 forward를 구현합니다.

```python
class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset: CorpusDataset,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.slot_labels_type = ["UNK", "PAD", "B", "I"]
        self.pad_token_id = 0
        self.bert_config = BertConfig.from_pretrained(
            self.config.bert_model, num_labels = len(self.slot_labels_type)
        )
        self.model = BertModel.from_pretrained(
            self.config.bert_model, config=self.bert_config
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.linear = nn.Linear(
            self.bert_config.hidden_size, len(self.slot_labels_type)
        )
		
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        x = outputs[0]
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
```
&nbsp;

training을 수행한 결과를 처리하는 부분을 작성합니다.

```python
def training_step(self, batch, batch_nb):

    input_ids, attention_mask, token_type_ids, slot_label_ids = batch
		
    outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    
    loss = self._calculate_loss(outputs, slot_label_ids)
    tensorboard_logs = {"train_loss": loss}
    
    return {"loss": loss, "log": tensorboard_logs}

def _calculate_loss(self, outputs, labels):
    active_logits = outputs.view(-1, len(self.slot_labels_type))
    active_labels = labels.view(-1)
    loss = F.cross_entropy(active_logits, active_labels)

    return loss

def _f1_score(self, gt_slot_labels, pred_slot_labels):
    return torch.tensor(
        f1_score(gt_slot_labels, pred_slot_labels), dtype=torch.float32
    )
```
&nbsp;

매 epoch마다 validation을 수행합니다.  
validation이 끝날 때마다 무언가를 하고 싶다면 validation_epoch_end()에 추가하면 됩니다.

```python
def validation_step(self, batch, batch_nb):
    input_ids, attention_mask, token_type_ids, slot_label_ids = batch

    outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    loss = self._calculate_loss(outputs, slot_label_ids)
    pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
        outputs, slot_label_ids
    )

    val_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

    return {"val_loss": loss, "val_f1": val_f1}

def validation_epoch_end(self, outputs):
    val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

    tensorboard_log = {
        "val_loss": val_loss,
        "val_f1": val_f1,
    }

    return {"val_loss": val_loss, "progress_bar": tensorboard_log}
```
&nbsp;

학습이 끝난 후 평가 할 test step도 동일하게 작성해줍니다.  
실제 추론한 결과를 리턴 받고 싶다면 test_epoch_end()의 return 값이 추가하면 됩니다.

```python
def test_step(self, batch, batch_nb):
    input_ids, attention_mask, token_type_ids, slot_label_ids = batch

    outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
        outputs, slot_label_ids
    )

    test_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

    test_step_outputs = {
        "test_f1": test_f1,
    }

    return test_step_outputs

def test_epoch_end(self, outputs):
    test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()

    test_step_outputs = {
        "test_f1": test_f1,
    }

    return test_step_outputs
```
&nbsp;

모델 생성을 완료하였습니다.  
최종 소스코드입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score

from utils import load_slot_labels
from dataset import CorpusDataset

class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset: CorpusDataset,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.slot_labels_type = ["UNK", "PAD", "B", "I"]
        self.pad_token_id = 0

        self.bert_config = BertConfig.from_pretrained(
            self.config.bert_model, num_labels=len(self.slot_labels_type)
        )
        self.model = BertModel.from_pretrained(
            self.config.bert_model, config=self.bert_config
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.linear = nn.Linear(
            self.bert_config.hidden_size, len(self.slot_labels_type)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        x = outputs[0]
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_labels)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_label_ids)
        pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
            outputs, slot_label_ids
        )

        val_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

        return {"val_loss": loss, "val_f1": val_f1}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

        tensorboard_log = {
            "val_loss": val_loss,
            "val_f1": val_f1,
        }

        return {"val_loss": val_loss, "progress_bar": tensorboard_log}

    def test_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_label_ids = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pred_slot_labels, gt_slot_labels = self._convert_ids_to_labels(
            outputs, slot_label_ids
        )

        test_f1 = self._f1_score(gt_slot_labels, pred_slot_labels)

        test_step_outputs = {
            "test_f1": test_f1,
        }

        return test_step_outputs

    def test_epoch_end(self, outputs):
        test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()

        test_step_outputs = {
            "test_f1": test_f1,
        }

        return test_step_outputs

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.config.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.config.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.config.eval_batch_size)

    def _calculate_loss(self, outputs, labels):
        active_logits = outputs.view(-1, len(self.slot_labels_type))
        active_labels = labels.view(-1)
        loss = F.cross_entropy(active_logits, active_labels)

        return loss

    def _f1_score(self, gt_slot_labels, pred_slot_labels):
        return torch.tensor(
            f1_score(gt_slot_labels, pred_slot_labels), dtype=torch.float32
        )

    def _convert_ids_to_labels(self, outputs, slot_labels):
        _, y_hat = torch.max(outputs, dim=2)
        y_hat = y_hat.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_labels_type)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.pad_token_id:
                    slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        return slot_pred_labels, slot_gt_labels
```

&nbsp;

## Reference
* <a href="https://github.com/PyTorchLightning/pytorch-lightning" target="_blank">PyTorch Lightning</a>
* <a href="https://github.com/SKTBrain/KoBERT" target="_blank">SKTBrain/KoBERT</a>
* <a href="https://github.com/monologg/KoBERT-NER" target="_blank">monologg/KoBERT-NER</a>

