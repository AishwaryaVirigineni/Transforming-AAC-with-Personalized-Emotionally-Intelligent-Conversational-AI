# 🎙️ EXPRESS ME: Personalized, Emotionally Intelligent AAC Chatbot

**EXPRESS ME** is a next-generation **Augmentative and Alternative Communication (AAC)** system powered by **Large Language Models (LLMs)**. Unlike traditional AAC tools that rely on static templates, this system enables **authentic, empathetic, and personalized conversations** — allowing users with speech impairments to express themselves in a more natural way.  

✨ Key Features:  
- 🧠 **Fine-tuned LLMs** (Mistral-7B-Instruct with QLoRA on DailyDialog & EmpatheticDialogues) for empathy, context, and personalization.  
- 📚 **Retrieval-Augmented Generation (RAG)** with autobiographical data for grounding responses in real-life context.  
- 🎛️ **Customizable Controls** for tone, length, and conversational intent (e.g., answering, asking, changing topic).  
- 💬 **Multimodal Output**: chatbot responds in **both text and audio**, enabling natural back-and-forth AAC interaction.  
- ⚡ **Optimized Inference** with CUDA + quantization (A100 GPU), reducing latency by 30%.  
- 🌐 **Streamlit Web App** for an intuitive, interactive user interface.  

---

## 📁 Project Structure  

### 🔹 `Finetune&Evaluate/`  
Contains all fine-tuning and evaluation notebooks.  
- `Finetune_Mistral7BInstruct_DailyDialog.ipynb` – Fine-tunes on **DailyDialog**.  
- `Finetune_Mistral7BInstruct_EmpatheticDialogue.ipynb` – Continues fine-tuning on **EmpatheticDialogues**.  
- `Evaluation_Mistral7BInstruct_EmpatheticDialogue.ipynb` – Evaluates BLEU, ROUGE, METEOR, BERTScore.  

### 🔹 `LLM_Evaluation_Report.pdf`  
Human + automatic evaluation results: relevance, fluency, personalization, sincerity.  

### 🔹 `PPT.pdf`  
Final project presentation slides.  

### 🔹 `Streamlit/`  
Streamlit web app for **real-time AAC interaction**.  
- `assistant_style.py` – Defines assistant tone/personality rules.  
- `PersonalNarrativePDF` – Autobiographical context for personalization.  
- `redis_saver.py` – Embeds & stores personal narratives in Redis for RAG.  

---

## 🚀 How It Works  

1. **Model Fine-Tuning**  
   - Start with Mistral-7B-Instruct.  
   - Stage 1: Fine-tune with DailyDialog for conversational quality.  
   - Stage 2: Further fine-tune with EmpatheticDialogues for empathy & emotional depth.  

2. **Personalization with RAG**  
   - User’s autobiographical narratives are embedded with SentenceTransformers.  
   - Redis Stack stores vectors for fast semantic retrieval.  
   - During chat, relevant narratives are retrieved & grounded in responses.  

3. **Interaction Flow**  
   - User inputs a message → Model generates personalized empathetic response.  
   - Output delivered in **both text and speech** for natural AAC experience.  
   - User can adjust **tone, response length, and intent** live in the Streamlit UI.  

---

## 📊 Results  

- **BERTScore F1**: 0.84 (vs baseline AAC response generators).  
- **Latency Reduction**: 30% via quantization + GPU optimization.  
- **Human Evaluation**: Significant improvements in **personalization, fluency, empathy, and contextual relevance**.  

---

## 🛠️ Tech Stack  

- **LLM**: Mistral-7B-Instruct + QLoRA  
- **Datasets**: DailyDialog, EmpatheticDialogues  
- **Frameworks**: PyTorch, Hugging Face Transformers, TRL  
- **RAG**: LangChain, Redis Stack, SentenceTransformers  
- **Web App**: Streamlit + Text-to-Speech (audio replies)  
- **Compute**: NVIDIA A100 GPU  

---

## 🎯 Impact  

This project demonstrates how **LLMs can transform AAC** tools into **empathetic, emotionally intelligent conversational partners** — enabling users to go beyond basic communication and **express personality, emotion, and intent** in real-time conversations.  

---

## 📜 Citation  

If you use this work, please cite:  
```
Virigineni, Aishwarya. EXPRESS ME: Transforming AAC with Personalized, Emotionally Intelligent Conversational AI. SUNY Buffalo, 2025.
```
