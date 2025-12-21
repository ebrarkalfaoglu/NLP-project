# NLP-project

## Progression

### Week 1: Base Model

In the first week, I downloaded the base model, Qwen2.5-Coder-1.5B, and tested it with a prompt. Here is one successful answer:

<img width="1706" height="643" alt="image" src="https://github.com/user-attachments/assets/7a991663-9e75-44c8-a5f8-c68911150437" />

### Week 2: Dataset Analysis

In the second week, we downloaded and prepared the required datasets for fine-tuning.

<img width="1302" height="682" alt="image" src="https://github.com/user-attachments/assets/b76453e5-b3ab-45e0-856a-6d19de111205" />


The datasets are now ready for fine-tuning the model.

### Week 3 Fine-Tuning with LoRA (DEEP & DIVERSE)

In the third week, we fine-tuned the base model Qwen2.5-Coder-1.5B using LoRA (Low-Rank Adaptation) on two different datasets: DEEP and DIVERSE.

Both trainings started from the same base model and were conducted independently, as required.
Only the solution field of each dataset was used during training.

#### DEEP Model Training

- Fine-tuned using the DEEP dataset
- Focuses on more structured and consistent solution patterns
- LoRA was applied to reduce memory usage and training cost
- Training loss was monitored throughout the process:

<img width="1200" height="600" alt="loss_graph" src="https://github.com/user-attachments/assets/820efaff-c30e-47d8-940d-d475fbf24984" />

#### DIVERSE Model Training

- Fine-tuned using the DIVERSE dataset
- Contains more varied and heterogeneous solution examples
- Trained separately from the DEEP model using the same base model
- Loss values were tracked to analyze training stability:

<img width="1200" height="600" alt="loss_graph_diverse" src="https://github.com/user-attachments/assets/e7f84609-c57f-4e65-b5c9-16ebb74aa1f8" />


#### Observations

- The DEEP-LoRA model shows a steady and continuous decrease in both training and validation loss.
- Training and validation curves follow a similar trend, indicating stable learning and good generalization.
- No signs of overfitting were observed during DEEP training.

- The DIVERSE-LoRA model demonstrates an initial rapid decrease in training loss, followed by an early plateau.
- Validation loss for the DIVERSE model stabilizes early, suggesting limited additional generalization after a certain point.
- Despite minor fluctuations, training remains stable without overfitting.

Overall, the DEEP dataset enables more effective and consistent learning, while the DIVERSE dataset provides stable but more limited improvements due to its heterogeneous structure.
At the end of Week 3, we obtained two independently fine-tuned LoRA models, which are ready for further evaluation and benchmarking in the next stage of the project.



