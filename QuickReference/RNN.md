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
4. Backpropagation through time is used to back propagate the losses.

## What are the different architectures of RNN?

[Refer Andrej Karpathy's Blog](<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>)

[Refer Slide Share on RNN](<https://www.slideshare.net/ananth/recurrent-neural-networks-lstm-and-gru>)


