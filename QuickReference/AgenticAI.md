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
    * To make the LLMs more useful, researchers introduced **Supervised Fine-Tuning (SFT) and Reinforced Learning with Human Feedback (RLHF)**.
         * **SFT** involved training the model with explicit instruction-following data. Researchers curated datasets where human responses were directly associated with specific queries. This helped Models understand the structure of proper responses.
         * **RLHF** further refined these responses by incorporating human preferences. AI models were fine-tuned based on which response humans rated as better, ensuring more coherent and relevant replies.

   * [ADVANCES AND CHALLENGES IN FOUNDATION AGENTS:
FROM BRAIN-INSPIRED INTELLIGENCE TO EVOLUTIONARY, COLLABORATIVE, AND SAFE SYSTEMS](https://arxiv.org/pdf/2504.01990)
      * [Github location](https://github.com/FoundationAgents/awesome-foundation-agents)
        

## About Agents in AI
As LLMs became more capable, new challenges arose. Users started expecting AI to handle factual and real-time queries, such as:

"What is the square root of 14.52321?"
"What is the current GDP of India?"

These were difficult for the models because:
1. LLMs are **not calculators** and do not inherently perform precise mathematical operations
2. LLMs rely **on pre-trained knowledge and do not have real-time access to external data sources** like the latest economic reports or stock market updates. 

To address these limitations, AI developers, began integrating **function-calling capabilities into models** (or their workflows). Function calling allows an AI model to invoke external tools to **retrieve up-to-date information** or perform specific tasks. 

This enhancement allowed LLMs to provide **accurate, up-to-date, and computationally precise responses**. 



