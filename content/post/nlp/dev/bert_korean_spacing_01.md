---
title: "BERT를 이용한 한국어 띄어쓰기 모델 만들기 - 01. 데이터 준비"
date: 2021-01-07 23:11:00 +0800
categories: [nlp]
tags: [bert, spacing, 띄어쓰기]
toc: true
---

Pretrained BERT, PYTORCH를 이용해 한국어 띄어쓰기 모델을 만들어 보겠습니다.

## 한국어 데이터셋
한국어 띄어쓰기 모델을 학습하기 위해서는 띄어쓰기가 잘 되어 있는 데이터셋이 필요합니다.  
세종 코퍼스, 모두의 말뭉치 처럼 오픈 데이터셋도 있고, 네이버 뉴스 같은 것을 직접 크롤링 할 수도 있습니다.  
이번에는 최근에 공개된 모두의 말뭉치 중에서 신문 말뭉치를 이용해보겠습니다.

> <a href="https://corpus.korean.go.kr" target="_blank">모두의 말뭉치</a>는 말뭉치 신청 후 승인이 되면 다운로드가 가능합니다.  


<br/>

## 데이터셋 전처리
말뭉치를 다운로드 받으면 제목, 기자, 날짜, 토픽, 내용 등이 json 형태로 되어 있어 파싱 후 사용해야 됩니다.  
직접 파싱을 할 수도 있으나, 감사하게도 <a href="https://github.com/ko-nlp/Korpora" target="_blank">Korpora</a>라는 라이브러리를 만들어 주셔서 편하게 데이터셋을 이용할 수 있습니다.

&nbsp;

그럼 모두의 말뭉치 데이터셋을 로드해보겠습니다.
```python
from Korpora import Korpora

# root_dir에 말뭉치 경로를 입력
modu_news = Korpora.load("modu_news", root_dir="./data/nikl_newspaper")
```
>Korpora에서 여러가지 오픈 데이터셋을 이용할 수 있습니다. 데이터셋을 직접 다운로드도 가능하나, 모두의 말뭉치는 승인절차가 필요하므로 다운로드는 제공하지 않습니다.

&nbsp;

불러온 데이터를 출력해보겠습니다.
```shell
>> modu_news.train[0]
ModuNewsLight(document_id='NIRW1900000001.1', title='오마이뉴스 2009년 기사', paragraph='대통령, 시장 방문만 하지...',)

>> modu_news.get_all_texts()
['대통령, 시장 방문만 하지...', ...]
```

&nbsp;

신문기사이기 때문에 매우 정제된 데이터이나, 간단한 전처리를 해보겠습니다.

* 따옴표 제거
    * 내용에 큰/작은 따옴표가 굉장히 많기 때문에 따옴표만 제거해주겠습니다.
* 개행문자로 문장 분리
    * 문장들이 개행문자(\n)로 연결되어 하나의 str로 되어 있어, 개행문자로 문장을 분리하겠습니다.
* 문장 분리
    * 개행문자 없이 여러 문장이 하나의 str로 되어 있는 경우도 있어, 문장을 분리하겠습니다.
    * 마찬가지로 매우 감사하게 사용하고 있는 <a href="https://github.com/hyunwoongko/kss" target="_blank">kss</a> 라이브러리를 사용해 문장을 분리하겠습니다.
    
```python
import kss

splited_sentences = []
corpus = modu_news.get_all_texts()

for sentences in corpus:
    sentences = sentences.replace("'", "")
    sentences = sentences.replace("\"", "")
    sentences = sentences.split("\n")

    for sentence in sentences:
        splited_sentences.extend(kss.split_sentences(sentence))
```

```shell
>> splited_sentences
["대통령, 시장 방문만 하지 말고 실천해달라", "2008년의 마지막 새벽, 언론의 카메라는 서울 여의도를 향했다. 방송법 등 주요쟁점 법안이 상정될 국회 본회의장을 두고 여야 의원들의 전쟁을 기다리고 있었던 것.", ...]
```

&nbsp;

kss로 자르기 전 문장 개수가 2,693,991개 입니다.  
제가 가진 장비로 2,693,991 문장에 대해 전처리를 하면 3시간이 소요됩니다.  
실습을 위해서는 문장 개수를 줄여서 해도 되지만, 병렬처리를 통해 시간을 단축해보겠습니다.  
python multiprocessing을 사용하는 대신, 요즘 많이 사용하는 <a href="https://docs.ray.io/en/master/index.html" target="_blank">Ray</a>를 이용해보겠습니다.  


```python
import ray
import kss
from itertools import chain

ray.init()

@ray.remote
def split_sentences(corpus):
    splited_sentences = []

    for sentences in corpus:
        sentences = sentences.replace("'", "")
        sentences = sentences.replace("\"", "")
        sentences = sentences.split("\n")

        for sentence in sentences:
            splited_sentences.extend(kss.split_sentences(sentence))
    return splited_sentences

def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))

# 프로세스 개수
process_num = 30

# 프로세스 개수만큼 corpus 분리
sentences_chunk = list(chunker_list(corpus, process_num))

# ray를 이용해 멀티프로세싱
futures = [split_sentences.remote(sentences_chunk[i]) for i in range(process_num)]

# 결과를 1-d로 합치기
results = ray.get(futures)
results = list(chain.from_iterable(results))
```

&nbsp;

최종적으로 6,138,136 문장으로 분리 되었습니다.  

모델 학습을 위해 train, validation, test로 분리하겠습니다.

```python
from sklearn.model_selection import train_test_split

test_size = int(len(results) * 0.1)
train, val = train_test_split(results, test_size=test_size, random_state=111)
train, test = train_test_split(train, test_size=test_size, random_state=111)
```

&nbsp;

각 데이터셋을 파일로 저장 후 사용하겠습니다.

```python
def write_dataset(file_name, dataset):
    with open(
        os.path.join("data/nikl_newspaper", file_name), mode="w", encoding="utf-8"
    ) as f:
        for data in dataset:
            f.write(data)

write_dataset("train.txt", train)
write_dataset("val.txt", val)
write_dataset("test.txt", test)
```
&nbsp;

이상으로 한국어 띄어쓰기 모델 개발을 위한 데이터셋 준비를 마쳤습니다.

&nbsp;

## Reference
* <a href="https://corpus.korean.go.kr" target="_blank">모두의 말뭉치</a>
* <a href="https://github.com/ko-nlp/Korpora" target="_blank">Korpora</a>
* <a href="https://github.com/hyunwoongko/kss" target="_blank">kss</a>
* <a href="https://docs.ray.io/en/master/index.html" target="_blank">Ray</a>
