# About NLP

[TOC]



NLP is **Natural Language Processing**, a branch of **AI** that enables machines to understand human language.

The objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable.

NLP is considered one of the difficult problems in computer science as the human language deals with ambiguity, and imprecise characteristics which are difficult to represent in machines. 



## Embeddings

An embedding in NLP is a continuous representation of a discrete object (such as a word or a character) in a high-dimensional vector space, used as a feature in NLP models. The goal is to capture semantic and syntactic relationships between the objects being embedded.

For example, in a word embedding model, each word in a vocabulary is assigned a dense vector of fixed size, such that semantically similar words are represented by vectors that are close to each other in the vector space. For example, the vectors for the words *"dog" and "puppy"* might be close to each other, while the vectors for *"dog" and "car"* would be far apart.

## NLP 0.1

**Bag of Words**: In this model, the input (the corpus of text/document) is represented as a **bag** (set) of its words, disregarding the grammar and the order of the words.

One main application is to classify documentation based on the frequency of the words, which is used as a feature to train the classifier.

Another application is **spam filtering**.

## NLP 1.0 

### How semantic is captured in embeddings?

Semantic information is captured in embeddings through their relationships with other words in the corpus. The model learns the relationships between words based on the context in which they appear, such as the words that surround them, their co-occurrence patterns, and relationships with other words. This information is encoded in the vectors such that semantically similar words have similar vectors and are close to each other in the vector space.

---

### RNNs: LSTM and GRU

When the input is a text stream (aka **Sequential Data**), to understand the context, the current word depends on the previous words (state). In order to model to understand the context, the concept **RNN** (Recurrent Neural Network) is used. 

RNNs use some **memory** element to **remember** the context/dependencies. The model uses the **feedback loop** from the output to the input to remember. You can visualize it as **finite-state machines**. 



### What is the problem with RNN?

One of the main problems with RNNs is the vanishing gradient issue. In an RNN, gradients are backpropagated through time and can become very small or even zero as they move further back in time. This results in the model failing to update the parameters of the earlier time steps and leads to poor performance.

Another issue with RNNs is the limited ability to capture long-term dependencies in sequences. RNNs have a tendency to forget information over time, making it difficult to capture and maintain context over a long sequence.

To address these issues, variants of RNNs have been developed, such as long short-term memory (LSTM) and gated recurrent units (GRUs), which introduce gates to regulate the flow of information and gradients through the network, improving its ability to capture long-term dependencies.

---



The popular RNNs are **LSTM** (Long Short-Term memory) and **GRU** ( Gated Recurrent Unit).

Another flavor that seems to increase the accuracy of the prediction/classification based on the context, is **Bi-directional LSTM**. In this model, two LSTMs are involved, where one LSTM is trained from left to right (from start to end of the text corpus) and another one is trained from right to left (end to start of the text corpus). Finally, both the LSTMs outputs are used to predict or classify.

To increase the accuracy of the prediction/classification, even **CNN** (Convolutional Neural Network) is used along with **LSTM**. Note that in the CNNs, the **convolution** operation is nothing but the **feature extractors**. Usually, CNNs are used for image/video / audio processing. In the case of NLP, **convolutions** are used to understand the context.

**LSTMs are slow to train**.



### What is the problem with LSTM?

One issue with LSTMs is their computational and memory complexity. LSTMs have a larger number of parameters and a more complex structure than other types of RNNs, making them computationally expensive to train and use. Additionally, LSTMs require more memory to store the hidden state, which can be problematic for long sequences or large datasets.

Another issue with LSTMs is that they can still struggle with capturing long-term dependencies, especially for very long sequences or highly complex relationships. This can limit the effectiveness of LSTMs for certain NLP tasks, such as language modeling or machine translation, where long-term context is important.

Finally, LSTMs are subject to overfitting, especially when training on limited data, which can result in poor generalization performance on unseen data. Regularization techniques, such as dropout, can be used to mitigate overfitting, but may still result in suboptimal performance.



### What is the problem with GRU?

GRUs are simpler and faster than LSTMs but can still have limitations. One issue with GRUs is that they have a smaller number of parameters than LSTMs, which can result in a lower capacity to capture complex relationships in sequences.

Another issue with GRUs is the potential for underfitting, especially when training on limited data or with complex tasks. GRUs may not be able to capture all the relevant information in a sequence, leading to suboptimal performance.

Finally, GRUs are still susceptible to overfitting, which can result in poor generalization performance on unseen data. Regularization techniques, such as dropout, can be used to mitigate overfitting, but may still result in suboptimal performance.

In summary, GRUs provide a trade-off between computational efficiency and representational capacity, and their performance may depend on the specifics of the task and the size of the dataset.

---



## NLP 2.0 

## Attention Mechanism

The attention mechanism in NLP is a mechanism for dynamically focusing on different parts of the input sequence, allowing the model to **capture complex relationships** between elements in the sequence. It is used in deep learning models, such as attention-based models, to weigh the input features and dynamically attend to the most relevant information at each time step.

In an attention mechanism, **a set of attention weights** is generated for each time step, indicating the importance of each input element for the current time step. These **attention weights are used to weigh the input features and produce the final representation** for the current time step.

In NLP, **attention-based models** have been widely used for tasks such as **machine translation, text classification, and text generation**. For example, in machine translation, an attention-based model can attend to the most relevant words in the source sentence to generate the target translation.

Overall, the attention mechanism has proven to be effective in NLP tasks, allowing models to capture complex relationships in sequences and providing improved performance compared to traditional RNN and LSTM models.

## Transformers

**Transformers**: is an architecture that uses **multiple encoders and decoders** for language modeling. **Each encoder and decoder** in turn uses something called **self-attention models**. This model helps to solve sequential data with long-range dependencies ( think of it as long sentences). This architecture **does not** use **RNNs and Convolutions**.


![Evolution of Large Language Models](https://www.researchgate.net/profile/Ruixiang-Tang/publication/370224758/figure/fig1/AS:11431281153210098@1682361835770/The-evolutionary-tree-of-modern-Large-Language-Models-LLMs-traces-the-development-of.png)

Transformer models are based on paper - Attention is all about.
The transformer model has two main components:

      * Encoder 
        In a standard Transformer architecture, the encoder component is used to process the input sequence and generate a representation of it, which is then fed into the decoder component to generate the output sequence.  

      * Decoder
        In the Transformer architecture, the decoder component is responsible for **_generating the output sequence_** based on the **input sequence** and the **encoder-generated representation of it**.



 **BERT** (**Bi-directional Encoder Representations** from **Transformers**): It's a model (pre-trained with a text corpus) **from Google**. It's an encoder only model.

**GPT** (**Generative Pre-trained Transformer** ): It's a model based on the transformer architecture **from OpenAI**. It's a decoder only model.

**Transformer models are faster to train**.

Some Resources to get you started on Transformers:

* [Step by Step explaination of Transformer Architecture | YouTube Video - A.I. Hacker Michal Pai](https://youtu.be/4Bdc55j80l8)
* [Transformer Architecture : YouTube](https://www.youtube.com/watch?v=H39Z_720T5s&list=PLo2EIpI_JMQtNtKNFFSMNIZwspj8H7-sQ&index=5)

* [About GPT, ChatGPT etc.](https://lifearchitect.ai/chatgpt/)

Which is better - BERT or GPT?

## References

* [Intro to NLP : Article](https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32)
* [Understanding Attention in Deep Learning](https://towardsdatascience.com/attaining-attention-in-deep-learning-a712f93bdb1e)
* [About Transformers in NLP](http://jalammar.github.io/illustrated-transformer/)
* 

