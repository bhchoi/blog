---
title: "[논문리뷰] Attention Is All You Need"
date: 2021-03-03 20:11:00 +0800
categories: [nlp]
tags: [attentio, transformer]
toc: true
---

# Abstract

대부분의 sequence를 처리하는 model들은 RNN, CNN기반의 encoder-decoder model이고, 가장 성능이 좋은 model은 encoder와 decoder를 attention으로 연결한 모델이다.
recurrence와 convolution은 완전히 배제하고 attention만 이용한 transformer model을 제안한다.  
&nbsp;

# Introduction
LSTM, GRU와 같은 RNN은 sequence modeling과 transduction 문제에서 SOTA의 접근방법이었다. 그러나 RNN model은 sequence를 순차적으로 처리해야하는 제약으로 인해 병렬처리가 어렵고, 문장이 길어질 수록 메모리 부족 문제에 빠지게 된다. attention 메카니즘은 input, output의 거리에 상관없이 dependency를 모델링을 할 수 있지만, 대부분의 경우 RNN과 함께 사용된다. 본 논문에서는 RNN을 사용하지 않고 attention만 사용하여 학습 속도와 성능 모두 향상을 시켰다.  
&nbsp;

# Model Architecture

왼쪽 부분이 encoder이며, 오른쪽 부분이 decoder이다. encoder와 decoder 모두 stacked self-attention과 point-wise fully connected로 구성되어 있다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/figure_1.png"></p>  
&nbsp;

## Encoder and Decoder Stacks

### **Encoder**

encoder는 N(6)개의 동일한 layer로 구성되어 있고 각각의 layer는 2개의 sub-layer를 가지고 있다. 첫번째는 multi-head self-attention이며, 두번째는 position-wise fully connected feed-forward network이다. 두 sub-layer는 residual connection이 추가되고, layer normalization을 수행한다. 수식으로 표현하면 LayerNorm(x + Sublayer(x))와 같다. residual connection을 용이하게 하기위해 output dimension은 512로 정하였다.  
&nbsp;

### **Decoder**

decoder도 N(6)개의 동일한 layer로 구성되어 있고, encoder와 동일한게 2개의 sub-layer가 있고, 추가로 encoder의 output을 처리하는 sub-layer가 하나 더 존재한다. sub-layer는 residual connection으로 연결되며 layer normalization을 수행한다. sub-layer의 self-attention을 수정하여 각 position 뒤쪽의 position은 참조하지 못하도록 하였다. 이를 마스킹이라고 하는데, position i에 대한 prediction은 오직 i 이전 position의 output만 참조하게 된다.  
&nbsp;

## Attention

attention은 query와 key-value pair를 output으로 mapping 하는 것이다. query, key, value, output 모두 vector이다. output은 value의 weighted sum으로 계산된다.  
&nbsp;

### **Scaled Dot-Product Attention**

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/figure_2.png"></p>  

scaled dot-product attention의 input은 d_k dimension 크기의 query와 key, 그리고 d_v dimension의 value로 구성되어 있다. 하나의 query를 모든 key에 대해 dot-product를 계산하고, d_k의 제곱근으로 나눠주고, softmax를 적용하여 각 value에 대한 weight를 얻는다.

실제로 계산할때는 하나씩 계산하지 않고, 여러개를 하나의 Q, K, V matrix로 만들어 동시에 계산을 한다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_1.png"></p>  

일반적으로 additive attention과 dot-product attention가 많이 쓰인다. dot-product는 본 논문에서 d_k의 제곱근으로 나눠 주는 것을 제외하면 동일하다. 두 방법은 이론적으로는 복잡도가 비슷하지만, dot-product가 더 빠르고 효율적이다.

d_k의 제곱근으로 나눠주는 이유는, d_k가 큰 값을 가질때 dot product의 결과가 커지게 되고 softmax를 수행하면 gradient가 매우 작아지기때문에 학습이 잘 되지 않는다고 생각했기 때문이다.  
&nbsp;

### **Multi-Head Attention**

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/figure_3.png"></p>  

d_model dimension을 한번에 attention을 적용하지 않고, dimension을 h번 나누어서 적용한다.  query, key, value는 linear하게 h개로 projection된다. h개의 attention이 병렬로 수행되고 concat된다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_2.png"></p>  

논문에서는 8개의 head를 사용하였다. 각 vector들은 d_model의 크기에서 head만큼 나눠진 크기이다. 

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_3.png"></p>  

h번 수행하지만 head만큼 나누었기때문에, single-head attention과 계산량은 동일하다.  
&nbsp;

### **Applications of Attention in our Model**

transformer에서는 multi-head attention을 3가지 방법으로 사용한다.

- encoder-decoder attention layer에서, query는 이전 decoder layer에서 오고, key와 value는 encoder의 output이다. 이로 인해 decoder의 모든 위치에서 input sequence 전체를 참조할 수 있다.
- encoder는 self-attention layer를 포함한다. self-attention layer에서 key, value, query는 모두 동일하다. encoder의 각 위치에서 이전 encoder layer의 전체를 참조할 수 있다.
- decoder에서는 각 위치의 이전 위치까지만 참조할 수 있다. scaled dot-product attention에서 masking을 추가하여 구현하였다.  
&nbsp;

### **Position-wise Feed-Forward Networks**

encoder, decoder 모두 fully connected feed-forward network를 가지고 있다. 2개의 linear transformation으로 구성되어 있고, ReLU를 사용한다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_4.png"></p>  
&nbsp;

### **Embeddings and Softmax**

input, output token을 vector로 변환하기 위해 d_model dimension의 embedding을 사용한다. 또한 decoder output을 next-token probability로 변환하기 위해 linear transformation과 softmax를 사용한다. 2개의 embedding layer에서 동일한 weight matrix를 사용한다. weight를 √d_model를 곱해준다.  
&nbsp;

### **Positional Encoding**

rnn, cnn을 포함하고 있지 않기 때문에 sequence의 각 토큰의 순서정보가 필요하다. 이를 위해, positional embeddings를 넣어준다. positional embeddings의 크기는 d_model dimension이고, input embeddings와 더해서 사용한다.

positional encoding에는 여러 방법이 있었지만, sine, cosine 함수를 이용한 방법을 선택했다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_5.png"></p>  

pos는 position이며, i는 dimension이다. 이 방법을 선택한 이유는 relative position을 더 쉽게 학습할 수 있기 때문이다.  
&nbsp;

# Why Self-Attention

self-attention만 사용한 3가지 이유가 있다. 첫째는 layer의 연산 복잡도가 줄어든다는 점이고, 둘째는 병렬처리 가능한 연산량이 증가한다는 것이다. 셋째는 long range dependency를 더욱 잘 학습할 수 있다는 것이다.

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/table_1.png"></p>  

추가적으로 self-attention은 해석하기 쉬운 모델이다. 

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/figure_4.png"></p>  
&nbsp; 

# Training

## Training Data and Batching

450만개의 문장 쌍과 byte-pair로 인코딩된 37,000개의 토큰으로 구성된 WMT 2014 English-German dataset을 학습하였다.
또한 3600만개의 문장 쌍과 word-piece를 통해 32,000개의 토큰으로 구성된 WMT 2014 English-French도 사용하였다.
배치에는 25,000개의 토큰을 포함하는 문장 쌍으로 구성되어 있다.  
&nbsp;

## Hardware and Schedule

8개의 P100 GPU 장비를 사용하였다. base model은 각 step에 0.4초가 소요되었고, 총 100,000 step이나 12시간을 학습하였다. big model은 한 step에 1초가 소요되며, 총 300,000 step을 3.5일동안 학습하였다.  
&nbsp;

## Optimizer

optimizer는 Adam을 사용하였다. 

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_6.png"></p>  

<p align="center"><img src="/images/nlp/paper/attention_is_all_you_need/eq_7.png"></p>  

warmup_steps동안은 선형적으로 증가하고, 그후 각 step의 inverse square root에 비례하여 감소한다.
warmup_steps은 4000이다.  
&nbsp;

## Regularization

3가지의 regularization을 적용하였다. 

- sub layer의 output에 dropout 0.1 적용
- attention을 구한 후 dropout 0.1 적용
- Label Smoothing 0.1 적용