# Recurrent Neural Network (Sequential Models)

[TOC]

## What is Recurrent Neural Network (RNN)?

The Recurrent Neural Network (RNN) is a kind of neural network, which is designed to recognize the data's sequenential characteristics and use patterns to predict the next likely scenario.

They are commonly used in the following areas:

* Speech Recognition (speech to text translation; input sequence to output sequence)
* Natural Language Processing (NLP)
* Sentiment Classification (From movie review to number of stars)
* Image Captioning (sequence of video graps -> identify the activity)
* Name Entity Recognition (From the sequence of text -> identify the people)

## Can we use standard Feedforward NN to process sequence of text?

For the sequence of  text, the standard feedforward NN suffers for the following reasons:

* It doesn't share features learned across different positions of text.
* Input and Output can be of different length.

## How does RNN learn?

[Basic RNN Architecture](https://images.app.goo.gl/2F6VKAHsir2SMsaa7)

[RNN Tutorial](<http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>)

A Few things to note about RNN which are different from regular NN are:

1. The output not only depends on input but also the previous activation (state). 
2. The hidden state of RNN is also called as memory.
3. RNN shares the same parameter across all steps. Thus, reduces the number of parameters.
4. **Backpropagation through time (BPTT)** is used to back propagate the losses.

## What are the different architectures of RNN?

[Refer Andrej Karpathy's Blog](<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>)

[Refer Slide Share on RNN](<https://www.slideshare.net/ananth/recurrent-neural-networks-lstm-and-gru>)



## How the Vanishing  / Exploding Gradient resolved in RNN?

As the RNN has deeper NN structure, the gradients computed based on the loss / error needs to beback propagate to the input. 

When the gradient is small ( < 1), when they are back propagated to the input, the gradient may approach *zero / very negligible value* results in gradient vanishing. 

When the gradient is big ( > 1), when they are back propagaged, the gradient will become a big value, that is the gradient explodes. 

In both the scenarios, it affects the functionaly of NN as weights close to the inputs become either zero / huge number - ie., weight is **not appropriate**.

The vanishing / exploding gradient can be resolved in many ways. One of them is: **LSTM.**

**LSTM** has the following things:

* the direct memory pipeline throughout the network

* *forget gate*(sigmoid function) - to erase the old memory or not

* input gate(sigmoid + tanh function)

* Output gate(sigmoid function)

  



## Blogs

* [Chistopher Olah Github blog](<http://colah.github.io/posts/2015-08-Understanding-LSTMs/>)
* [RNN in Tensorflow](<https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html>)
* [WildML : RNN Tutorial](<http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>)
* [SlideShare: RNN](<https://www.slideshare.net/ananth/recurrent-neural-networks-lstm-and-gru>)
* [Kaggle - RNN Intro](<https://www.kaggle.com/honeysingh/intro-to-recurrent-neural-networks-lstm-gru>)



## Papers

* [Visualizing and Understanding Recurrent Networks | Andrej Karpathy ](<https://arxiv.org/pdf/1506.02078v2.pdf>)
* 