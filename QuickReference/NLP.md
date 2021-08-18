# About NLP

[TOC]



NLP is **Natural Language Processing** which is a branch of **AI** that enables machines to understand human language.

The objective of NLP is to read, decipher, understand and make sense of human languages in a manner that is valuable.

NLP is considered one of the difficult problem in computer science as the human language deals with ambiguity, imprecise characteristics which are difficult to represent in machines. 

## NLP 0.1

**Bag of Words**: In this model, the input (corpus of text / document) is represented as **bag** (set) of its words, disregarding the grammar and the order of the words.

One of the main application is to **classify** documentation based on the **frequency** of the words, which is used as feature to train the classifier.

Another application is **spam filtering**.

## NLP 1.0 - RNNs

When the input is a text stream (aka **Sequential Data**), to the understand the context, the current word depends on the previous words (state). In order to model to understand the context, the concept **RNN** (Recurrent Neural Network) is used. 

RNNs uses some **memory** element to **remember** the context / dependencies. The model uses the **feedback loop** from the output to input to remember. You can visualize as **finite state machines**. 

Following are the popular RNNs - **LSTM** (Long Short-Term memory) and **GRU** ( Gated Recurrent Unit).

Another flavour seems to increase the accuracy of the prediction / classification based on the context, is **Bi-directional LSTM**. In this model, two LSTMs are involved, where one LSTM is trained from left to right (from start to end of the text corpus) and another one is trained from right to left (end to start of the text corpus). Finally, both the LSTMs outputs are used to predict or classify.

To increase accuracy of the prediction / classification, even **CNN** (Convolutional Neural Network) is used along with **LSTM**. Note that in the CNNs, the **convolution** operation is nothing but the **feature extractors**. Usually, CNNs are used for image / video / audio processing. In the case of NLP, **convolutions** are used to understand the context.

**LSTMs are slow to train**.

## NLP 2.0 - Transformers

**Transformers**: is a architecture which uses **multiple encoder and decoder** for the language modeling. **Each encoder and decoder** in turn uses something called **self-attention models**. This model helps to solve sequential data with long range dependencies ( think of it as long sentences). This architecture **does not** use **RNNs and Convolutions**.

 **BERT** (**Bi-directional Encoder Representations** from **Transformers**): It's a model (pre-trained with a text corpus) **from Google**.

**GPT** (**Generative Pre-trained Transformer** ): It's a model based on tranformer artchitecure **from OpenAI**.

**Transformer models are faster to train**.

Which is better - BERT or GPT?

## References

* [Intro to NLP : Article](https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32)
* [Understanding Attention in Deep Learning](https://towardsdatascience.com/attaining-attention-in-deep-learning-a712f93bdb1e)
* [About Transformers in NLP](http://jalammar.github.io/illustrated-transformer/)
* 

