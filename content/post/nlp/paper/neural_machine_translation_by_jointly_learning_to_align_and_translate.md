---
title: "[논문리뷰] Neural Machine Translation by Jointly Learning to Align and Translate"
date: 2021-02-23 22:11:00 +0800
categories: [nlp]
tags: [seq2seq, attention]
toc: true
---

> <a href="https://arxiv.org/abs/1409.0473" target="_blank">https://arxiv.org/abs/1409.0473</a>

# ABSTRACT
encoder-decoder model의 고정된 길이의 vector는 성능향상에 bottleneck으로 작용한다고 생각한다.  
이를 개선 할 방법으로, 예측한 target word가 source sentence의 어떤 부분들과 연관이 있는지 자동으로 찾는 모델을 제안한다. 

&nbsp;

# INTRODUCTION

encoder-decoder 방식의 문제점은 source sentence를 고정된 크기의 vector에 압축하기때문에, 긴 문장에 대해서 잘 처리하지 못한다는 점이다. 특히 학습셋보다 더 큰 문장에 대해 더욱 성능이 좋지 않다.

이를 해결하기 위해 align과 translate을 함께하는 방법을 제안한다. 각 time step에서 모델은 번역된 word를 생성하고, source sentence에서 가장 연관이 있는 부분들을 찾는다. 그런 다음 모델은 source position에 기반한 context vector와 이전에 생성된 target word를 이용하여 target word를 예측한다.

기존 encoder-decoder 방식과 가장 크게 구별되는 점은 input sequence에 대해 하나의 고정된 길이의 vector로 표현하는 것이 아니라, vector sequence로 표현하고 decoding하는 동안 이 vector의 subset을 adaptively하게 선택한다는 점이다.

이를 통해 긴 문장도 잘 처리되는 것을 확인하였다.  

&nbsp;

# LEARNING TO ALIGN AND TRANSLATE

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/figure_1.png"></p>

&nbsp;

## DECODER: GENERAL DESCRIPTION

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/eq_1.png"></p>

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/eq_2.png"></p>

일반적인 encoder-decoder 구조와 달리 context vector c_i는 target word y_i에 고유한 값이다.

c는 annotation이라 불리는 h_1, ..., h_Tx 값에 의해 결정되며, h_i는 input sequence에서 i번째 word 주변에 focus된 정보를 포함하고 있다.

c_i는 h_i의 weighted sum으로 계산된다.

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/eq_3.png"></p>

weight값 a_ij는 softmax로 계산된다.

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/eq_4.png"></p>

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/eq_5.png"></p>

e_ij는 input의 j번째 주변과 output의 i번째가 얼마나 잘 match되는지에 대한 score이다.

alignment model a는 feedforward neural network로 만들어지며, 전통적인 MT에서와 달리 latent variable가 아닌 직접적으로 계산된다.

모든 annotation에 대해 weighted sum을 하는 것은 모든 가능한 alignment에 대해 기대값을 계산하는 것으로 이해할 수 있다.

a_ij를 target word y_i가 source word x_j일 확률이라고 하자. 그러면 i번째 context vector c_i는 모든 annotation에 대한 기대값이다.

next state s_i를 결정하고 y_i를 생성할 때, 확률 a_ij 혹은 energy e_ij는 이전 hidden state s_i-1을 이용하여 annotation h_j의 중요성을 반영한다.

이러한 메카니즘을 attention이라고 한다. decoder에서 attention을 수행함으로써, encoder가 source sentence에 대해 모든 정보를 고정된 크기의 vector에 담아야 되는 부담을 덜어준다.

&nbsp;

## ENCODER: BIDIRECTIONAL RNN FOR ANNOTATING SEQUENCES

일반적인 RNN은 첫번째 글자부터 마지막 글자까지 순차적으로 읽는다. 이전 단어와 다음 단어 모두에 대한 annotation을 담기 위해 bidirectional RNN을 제안한다.

BiRNN은 forward와 backward로 구성되어있다. forward는 순방향으로 input sequence를 읽어서 hidden state를 계산하고, backward는 역방향으로 읽고 hidden state를 계산한다.

forward와 backward를 concat하여 annotation을 만든다. 그래서 annotation에는 이전 단어와 다음 단어의 정보를 모두 가지고 있게된다.

&nbsp;

# EXPERIMENT SETTINGS

## DATASET

English-to-Franch corpus인 WMT'14를 사용하여 평가하였다. 총 850M개의 word로 구성되있으나, 348M개의 word만 추출하여 학습하였다. 토큰화를 통해 가장 빈번하게 사용되는 30,000개의 단어를 선택하여 학습하였고, 포함되지 못한 단어는 [UNK]으로 치환하여 학습하였다. 그 외 전처리나 lowercasing, stemming은 사용하지 않았다.

&nbsp;

## MODELS

2가지의 모델을 학습하여 비교하였다. RNN Encoder-Decoder(RNNencdec, Cho et al. 2014a)와 본 논문에서 제안한 모델인 RNNsearch이다. 각 모델은 sentence의 길이를 30으로 제한, 50으로 제한하는 방법 2가지로 학습하였다.

RNNencdec의 encoder와 decoder는 각각 1,000개 hidden unit을 가지고 있다. RNNsearch의 encoder는  forward와 backward 각각 1,000개의 hidden unit을 가지고 있고, decoder도 1,000개의 hidden unit을 가진다. 각 target word 예측을 위해 single maxout hidden layer를 포함한 multilayer network를 사용한다.

Adadelta를 이용하여 minibatch SGD를 계산한다. minibatch size는 80이다.

학습 후 beam search를 이용하여 최대 확률의 translation을 구하였다.

&nbsp;

# RESULTS

## QUANTITATIVE RESULTS

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/table_1.png"></p>

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/figure_2.png"></p>

RNNsearch가 RNNencdec의 성능을 능가하였으며, 특히 RNNsearch-50은 긴 문장에서도 성능 하락이 없다는 것을 보여주었다.

&nbsp;

## QUALITATIVE ANALYSIS

### ALIGNMENT

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/figure_3.png"></p>

<p align="center"><img src="/images/nlp/paper/neural_machine_translation_by_jointly_learning_to_align_and_translate/figure_4.png"></p>

annotation weight값 a_ij를 visualizing한 모습을 보면, source sentence의 word가 target sentence의 어떤 word와 연관성이 있는지 알 수 있다.