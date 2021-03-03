---
title: "BERT를 이용한 한국어 띄어쓰기 모델 만들기 - 04. 학습"
date: 2021-01-14 20:11:00 +0800
categories: [nlp]
tags: [bert, spacing, 띄어쓰기]
toc: true
---

## 학습

이제 학습을 수행하는 부분을 작성하겠습니다.  
이전에 작성한 config yaml을 불러옵니다. config는 <a href="https://github.com/omry/omegaconf" target="_blank">OmegaConf</a>를 이용하여 불러옵니다.  


```python
from omegaconf import OmegaConf

config = OmegaConf.load("config/train_config.yaml")
```
&nbsp;

dataset과 model을 생성합니다.

```python
dataset = {}
dataset["train"] = CorpusDataset(
    config.train_data_path, preprocessor.get_input_features
)
dataset["val"] = CorpusDataset(
    config.val_data_path, preprocessor.get_input_features
)
dataset["test"] = CorpusDataset(
    config.test_data_path, preprocessor.get_input_features
)

bert_finetuner = SpacingBertModel(config, dataset)
```
&nbsp;

logging과 학습에 사용할 callback을 작성합니다.

- checkpoint_callback  : 매 epoch마다 validation loss를 계산해 loss가 줄어드는 경우 checkpoint를 저장합니다.
- early_stop_callback : validation loss가 더 이상 줄어들지 않으면 학습을 종료합니다.

```python
logger = TensorBoardLogger(
    save_dir=os.path.join(config.log_path, config.task), version=1, name=config.task
)

checkpoint_callback = ModelCheckpoint(
    filepath="checkpoints/"+ config.task + "/{epoch}_{val_loss:35f}",
    verbose=True,
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    prefix="",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=3,
    verbose=False,
    mode="min",
)
```
&nbsp;

마지막으로 학습을 수행하고 테스트를 진행합니다.  
PyTorch Lightning은 distributed training을 지원하기때문에 DistributedDataParallel을 이용해 학습을 하겠습니다.

```python
trainer = pl.Trainer(
    gpus=config.gpus,
    distributed_backend=config.distributed_backend,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=early_stop_callback,
    logger=logger,
)

trainer.fit(bert_finetuner)
trainer.test()
```
&nbsp;

## 결과

### 데이터셋

모두의 말뭉치 뉴스 데이터 전체를 학습하기에 시간이 오래걸려서 일부 데이터로 학습을 진행하였습니다.

- train : 100,000건
- val : 10,000건
- test : 10,000건
&nbsp;

### 그래프
![train_loss](/images/nlp/bert_spacing/bert_spacing_train_loss.png)  
![val_loss](/images/nlp/bert_spacing/bert_spacing_val_loss.png)  
![val_f1](/images/nlp/bert_spacing/bert_spacing_val_f1.png)

&nbsp;

### 성능

- F1

    ```
    report:            precision    recall  f1-score   support
                    B       0.97      0.96      0.97    120963
                    I       0.91      0.91      0.91    112427
            micro avg       0.94      0.94      0.94    233390
            macro avg       0.94      0.94      0.94    233390
    ```

- accuracy : 0.5294

&nbsp;

### 결과 예시

```
gt    : 특별한 일들이 생겨나고 있다.
pred  : 특별한 일들이 생겨나고 있다.
```

```
gt    : 몸싸움과 반격에 능한 차두리의 오버래핑은 이란의 강한 압박을 뚫고 주도권을 우리에게 유리하게 잡는데 필수적이다.
pred  : 몸싸움과 반격에 능한 차두리의 오버래핑은 이란의 강한 압박을 뚫고 주도권을 우리에게 유리하게 잡는데 필수적이다.
```

```
gt    : 2003년에 비교해 2010년 한국 사회는 어떤 모습인지?
pred  : 2003년에 비교해 2010년 한국사회는 어떤 모습인지?
```

```
gt    : 강원도 태백의 해발 935m인 삼수령 마루에 적혀있는 글이다.	
pred  : 강원도 태백의 해발 935m인 삼수령마루에 적혀 있는 글이다.	
```

&nbsp;

## Reference
* <a href="https://github.com/omry/omegaconf" target="_blank">OmegaConf</a>
* <a href="https://github.com/PyTorchLightning/pytorch-lightning" target="_blank">PyTorch Lightning</a>