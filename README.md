# NLP-project

## Progression

### Week 1: Base Model

In the first week, I downloaded the base model, Qwen2.5-Coder-1.5B, and tested it with a prompt. Here is one successful answer:

<img width="1706" height="643" alt="image" src="https://github.com/user-attachments/assets/7a991663-9e75-44c8-a5f8-c68911150437" />

### Week 2: Dataset Analysis

In the second week, we downloaded and prepared the required datasets for fine-tuning.

<img width="1302" height="682" alt="image" src="https://github.com/user-attachments/assets/b76453e5-b3ab-45e0-856a-6d19de111205" />


The datasets are now ready for fine-tuning the model.

### Week 3

This week, we converted both the Deep and Diverse datasets into JSONL format to match the instruction–output structure required by our LoRA training pipeline, ensuring efficient streaming and consistent formatting. After preparing the datasets, we configured the LoRA setup using the parameters listed above to balance training efficiency and model performance while keeping GPU memory usage low.
- r = 32
- lora_alpha = 64
- lora_dropout = 0.05
- target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
We then trained the model first on the Deep dataset to help it learn structured formatting and consistent input/output behavior, and subsequently on the Diverse dataset to improve its generalization and ability to handle a broader range of problem types. Together, these steps prepared the model for the final stage of fine-tuning on our custom topic, “Authors and Their Works.”

Here is a screenshot:

<img width="546" height="72" alt="image" src="https://github.com/user-attachments/assets/ec75c4ac-64a6-4823-9512-5ae330af7b97" />

I tested model with a that prompt:

<img width="1361" height="581" alt="image" src="https://github.com/user-attachments/assets/5bc259ed-2286-4661-bfa5-a7fbe77870da" />


