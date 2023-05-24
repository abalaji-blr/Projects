# Transformers - Large Language Models (LLM)

## About Transformers

**Transformer** is an architecture that uses **multiple encoders and decoders** for language modeling. **Each encoder and decoder** in turn uses something called **self-attention models**. This model helps to solve sequential data with long-range dependencies ( think of it as long sentences). This architecture **does not** use **RNNs and Convolutions**.

![Evolution of Large Language Models](https://www.researchgate.net/profile/Ruixiang-Tang/publication/370224758/figure/fig1/AS:11431281153210098@1682361835770/The-evolutionary-tree-of-modern-Large-Language-Models-LLMs-traces-the-development-of.png)

Transformer models are based on paper - Attention is all about.
The transformer model has two main components:

      * Encoder 
        In a standard Transformer architecture, the encoder component is used to process the input sequence and generate a representation of it, which is then fed into the decoder component to generate the output sequence.  

      * Decoder
        In the Transformer architecture, the decoder component is responsible for **_generating the output sequence_** based on the **input sequence** and the **encoder-generated representation of it**
