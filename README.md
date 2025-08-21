
---

## 📁 Project Structure

### 🔹 `Finetune&Evaluate/`
Contains all fine-tuning and evaluation notebooks.

- `Finetune_Mistral7BInstruct_DailyDialog.ipynb`  
  Fine-tunes the base Mistral-7B-Instruct model on the DailyDialog dataset.

- `Finetune_Mistral7BInstruct_EmpatheticDialogue.ipynb`  
  Continues fine-tuning on the EmpatheticDialogues dataset (starting from the DailyDialog-tuned checkpoint).

- `Evaluation_Mistral7BInstruct_EmpatheticDialogue.ipynb`  
  Evaluates the performance of the fine-tuned model using BLEU, ROUGE, METEOR, and BERTScore.

- `LLM_Evaluation_Report.pdf`  
  Results of LLM-based human evaluation on relevance, sincerity, personalization, etc.

- `PPT.ppt.pdf`  
  Final project presentation slides.

---
### 🔹 `LLM_Evaluation_Report.pdf`  
  Results of LLM-based human evaluation on relevance, sincerity, personalization, etc.

### 🔹 `PPT.pdf`  
  Final project presentation slides.


### 🔹 `Streamlit/`
Houses the Streamlit web app for real-time personalized AAC interaction using the fine-tuned model and retrieved personal narratives.


  - `assistant_style.py`
  Defines the assistant’s personality and tone styling rules for consistent output generation.

  - `PersonalNarrativePDF`
  A PDF document containing autobiographical context and personalized prompts for the AAC user.

  - `redis_saver.py`
  Embeds and stores autobiographical sentences into Redis for retrieval-augmented generation.

  ---

