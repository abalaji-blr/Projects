# Agentic AI

## Transformer Archiecture
* Universal Approximation Theorem
    * [Let's not reinvent](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-known-operator-learning-part-2/)
    * [Deep mind Paper](https://www.deep-mind.org/2023/03/26/the-universal-approximation-theorem/)
    * Neural Networks are universal approximations!
 
* Transformer Architecture
```
[ Input Text ]
       |
       v
[ Token Embedding ]
       |
       v
[ Positional Encoding ]
       |
       v
╔════════════════════╗
║   Encoder Block    ║  ← Repeated N times
║ ┌───────────────┐ ║
║ │ Self-Attention│ ║
║ └───────────────┘ ║
║ ┌───────────────┐ ║
║ │ Feedforward   │ ║
║ └───────────────┘ ║
╚════════════════════╝
       |
       v
[ Encoder Output ]
       |
       v
╔════════════════════╗
║   Decoder Block    ║  ← Repeated N times
║ ┌────────────────┐ ║
║ │ Masked Self-Att│ ║
║ └────────────────┘ ║
║ ┌────────────────┐ ║
║ │ Cross-Attention│ ║  ← Attends to Encoder Output
║ └────────────────┘ ║
║ ┌───────────────┐ ║
║ │ Feedforward   │ ║
║ └───────────────┘ ║
╚════════════════════╝
       |
       v
[ Output Tokens (Generated Text) ]
```

- The **Encoder** processes the input and builds a representation.
- The **Decoder** uses that representation to **generate output**, one token at a time.
- **Self-Attention** lets each word focus on others in the input.
- **Cross-Attention** helps the decoder focus on relevant parts of the input.

  * **Chinchilla Scaling Law** - if we train larger models over more data, we get better results
  * Supervised Fine Tuning (SFT)
  * 
  
## Research Papers / Reference Materials / Videos

* [Tokenization - Andrej Karapathy](https://www.youtube.com/watch?v=zduSFxRajkE)
* [Emergent Abilities of LLM](https://arxiv.org/pdf/2206.07682)
    
    *  Emergent abilities are behaviors that appear suddenly at **certain model scales** and are not predictable from smaller models.
    *  These abilities emerge due to **increased model size, training data, and the complexity of the learning landscape**.
    * The paper provides strong evidence that these abilities **arise non-linearly,** which has deep implications for the design, evaluation, and safety of large AI systems.

## About Agents in AI

