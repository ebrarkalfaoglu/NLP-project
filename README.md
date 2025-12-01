# NLP-project

## Progression
### Week 1

In the first week, I downloaded the base model, Qwen2.5-Coder-1.5B, and tested it with a prompt. Here is one successful answer:

<img width="1706" height="643" alt="image" src="https://github.com/user-attachments/assets/7a991663-9e75-44c8-a5f8-c68911150437" />

## Week 2: Dataset Analysis

In the second week, we downloaded and prepared the required datasets for fine-tuning.

- **Datasets used**: Deep (5K examples) and Diverse (5K examples)
- **Fields extracted**:
  - `solution`: code-only, used for training
  - `output`: reasoning traces (<think> tags), optional for reasoning enhancement
- **Total examples collected**:
  - Solutions: 10,000
  - Outputs: 10,000

The datasets are now ready for fine-tuning the model.

