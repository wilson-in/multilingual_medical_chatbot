📘 MediChat AI – Multilingual Medical Support Chatbot
A LoRA-Fine-Tuned FLAN-T5 Medical Question Answering System with Multilingual Translation Capabilities
1. Introduction

Access to trustworthy medical information is a global challenge, particularly in multilingual communities where health content is often available only in English. Millions of users search for medical help in regional languages, creating a gap between healthcare knowledge and linguistic accessibility.

MediChat AI addresses this problem through an integrated system that:

Accepts health-related questions in any language

Detects the input language automatically

Translates the query to English

Generates an accurate, medically grounded response using a LoRA-fine-tuned FLAN-T5-Base model

Translates the answer back to the user's language

Presents the result through a simple, intuitive Streamlit chatbot interface

The project blends advanced natural language processing, medical-domain training, and multilingual translation pipelines to produce a real-time medical information assistant.

⚠️ Important Medical Disclaimer
MediChat AI provides general health information and must not be used for diagnosis, medical decision-making, or emergencies.

2. Project Objectives
Primary Objective

Develop a scalable and reliable multilingual medical chatbot that can interpret, translate, analyze, and respond to health-related queries.

Secondary Objectives

Build a robust medical reasoning model using FLAN-T5 base fine-tuned with LoRA.

Integrate MarianMT translation pipelines for multilingual support.

Ensure lightweight, efficient deployment using HuggingFace Spaces.

Maintain high quality, safe, and user-appropriate information delivery.

Follow best practices in data preprocessing, fine-tuning, documentation, and version control.

3. Skills, Technologies & Tools
Technical Skills Gained

Natural Language Processing (NLP)

HuggingFace Transformers Architecture

Dataset Cleaning & Medical Text Preprocessing

Fine-Tuning FLAN-T5 using PEFT LoRA

Streamlit Frontend Development

Real-time Translation Pipelines

API Integration

Git & GitHub Version Control

Cloud Deployment (HuggingFace Spaces, AWS)

Libraries & Frameworks

transformers

datasets

peft (LoRA fine-tuning)

langdetect, fasttext

streamlit

sentencepiece

accelerate

torch

4. System Architecture (Elaborated)

The system is designed as a modular NLP pipeline to ensure maintainability, scalability, and clarity.

4.1 High-Level Architecture Flow
User Input (Any Language)
            |
            V
  Language Detection (utils.py)
            |
            V
 Translation to English (translator.py)
            |
            V
Medical Reasoning Model (FLAN-T5 LoRA)
            |
            V
 Generate Answer in English
            |
            V
Translate Back to User Language
            |
            V
Streamlit UI Output (app.py)


Each component is modular, replaceable, and independently testable.

4.2 Component Breakdown
A) Streamlit UI (app.py)

Clean, responsive chat interface.

Message bubbles styled for readability.

Displays original & translated answers.

Sidebar for model selection (FLAN-T5, BioGPT, custom fine-tuned model).

Real-time chat history & query logs.

Error handling and disclaimers integrated into UI.

B) Language Detection Module (utils.py)

Uses:

langdetect for quick predictions

FastText HF Model for fallback and robustness
https://huggingface.co/facebook/fasttext-language-identification

This hybrid setup ensures high accuracy for regional languages.

C) Translation Engine (translator.py)

Uses Helsinki-NLP MarianMT models for multilingual translation.

Examples:

Language	To English	Back Translation
Hindi	https://huggingface.co/Helsinki-NLP/opus-mt-hi-en
	https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

Tamil	https://huggingface.co/Helsinki-NLP/opus-mt-ta-en
	https://huggingface.co/Helsinki-NLP/opus-mt-en-ta

Telugu	https://huggingface.co/Helsinki-NLP/opus-mt-te-en
	https://huggingface.co/Helsinki-NLP/opus-mt-en-te

The system caches pipeline instances for efficiency.

D) Medical Model Handler (medical_model.py)
Exact Base Model Used
google/flan-t5-base


https://huggingface.co/google/flan-t5-base

Why FLAN-T5-Base?

Instruction tuned → ideal for Q&A tasks

Lightweight (250M params) → deployable on HuggingFace Spaces

High accuracy after LoRA fine-tuning

Faster inference vs 7B models

Fine-Tuned Version (LoRA Adapter)

Your notebook flant5_lora.ipynb fine-tunes FLAN-T5-Base using:

PEFT LoRA adapter

Rank reduction for efficient training

Mixed dataset supervision (PubMedQA, MedQuAD, ChatDoctor)

Multi-epoch training on cleaned biomedical Q&A format

This creates your customized model:

<your-username>/medichat-flan-t5-base-lora


The app loads this adapter for inference.

5. Datasets Used 

Your notebook uses a curated list of safe, high-quality medical datasets, defined in SAFE_CANDIDATES.

Here is the detailed explanation for each dataset:

5.1 PubMedQA (All 3 Configurations)

https://huggingface.co/datasets/qiaojin/PubMedQA

Configs Used:

pqa_labeled

pqa_artificial

pqa_unlabeled

Purpose:

Medical research Q&A dataset based on PubMed clinical studies.
Used for training the model to answer research-backed medical questions.

5.2 MedQuAD & Variants

A cluster of high-quality medical datasets focused on FAQ-style medical Q&A.

Datasets Included:

lavita/MedQuAD
https://huggingface.co/datasets/lavita/MedQuAD

Tonic/medquad
https://huggingface.co/datasets/Tonic/medquad

keivalya/MedQuad-MedicalQnADataset
https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

Amirkid/MedQuad-dataset
https://huggingface.co/datasets/Amirkid/MedQuad-dataset

Purpose:

Organized clinical Q&A

Covers diseases, symptoms, diagnoses, medications

Provides medically grounded responses

5.3 ChatDoctor Conversational Sets

ChatDoctor-HealthCareMagic-100k
https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k

ChatDoctor-iCliniq
https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq

Purpose:

Natural conversations between doctors and patients

Helps model learn real-world medical dialogue style

Good for symptom explanation & conversational flow

6. Fine-Tuning Process (Deeply Elaborated)

Your notebook performs LoRA-based fine-tuning on google/flan-t5-base.

6.1 Preprocessing
Tasks performed:

Merge multiple datasets

Clean text (remove HTML, noisy characters, duplicates)

Normalize terminology (e.g., "BP" → "Blood Pressure")

Convert all data into a unified structure:

{
  "instruction": "medical question",
  "input": "",
  "output": "medical answer"
}

6.2 Tokenization

FLAN-T5 tokenizer:

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


Tokenization includes:

Truncation to max token limit (512)

Padding for batching

Special tokens for instruction tuning

6.3 LoRA Training Details

Parameter-efficient fine-tuning using PEFT:

Rank: 8–16

Alpha: 16–32

Dropout: 0.05

Only ~0.3% of model parameters updated

Achieves high accuracy on limited GPU resources

Advantages:

Extremely memory-efficient

Fast training

Produces high-quality, domain-adapted model

6.4 Model Output Formatting

A controlled prompt template is used:

You are a medical assistant. Answer clearly and safely.
Question: {user_question}
Answer:


This ensures answer consistency.

7. Application Logic (Step-by-Step)
1. User types a question in any language
2. System detects the language
3. Query translated → English
4. Medical LLM processes the cleaned input
5. Answer translated back → original language
6. Streamlit displays answer + optional English version

This offers universal accessibility while maintaining medical accuracy.

8. Deployment Guide
8.1 Hugging Face Spaces

Create a new Space → Streamlit

Upload:

app.py

translator.py

medical_model.py

utils.py

requirements.txt

medichat-flan-t5-base-lora model folder (optional if hosted on HF Hub)

HF automatically runs:

streamlit run app.py --server.port 7860 --server.address 0.0.0.0

8.2 AWS Deployment

Use EC2 or App Runner

Build Docker image with Streamlit

Expose port 7860

Configure GPU instance if using large models

9. Safety, Ethics, and Error Handling
Safety:

Refuses non-medical questions

Warns users about diagnosis limitations

Adds disclaimers to each response

Ensures medical information remains general and safe

Error Handling:

Catches OOM errors

Times out gracefully

Logs exceptions to error_logs/

Detects translation failures and retries fallback model

10. Evaluation Metrics

Your project is evaluated based on:

Modular code structure

Maintainability & readability

Proper GitHub versioning

Data quality & preprocessing

Model performance on sample cases

Quality of documentation

Deployment success

Demo video on LinkedIn

11. Deliverables

Complete source code

Fine-tuning notebook flant5_lora.ipynb

README documentation

Requirements file

Trained model (LoRA adapter)

Demo video link

12. Final Notes

This project combines:

Real-time translation

Domain-specific LLM reasoning

Efficient LoRA fine-tuning

Multi-dataset medical QA training

Professional-grade UI design