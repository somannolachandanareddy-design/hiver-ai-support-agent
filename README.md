#Hiver AI Support Agent

An AI-powered customer support agent built using the **Customer Support on Twitter (TWCS)** dataset. The system learns from historical support conversations to classify customer issues, retrieve similar resolved cases, generate grounded responses, and decide whether to auto-handle or escalate to a human.

## Architecture

```text
Customer Message
       ↓
Intent Classification
       ↓
Similar Conversation Retrieval
       ↓
LLM Response Generation
       ↓
Auto-handle / Escalate
```

## Problem

The goal is not just to generate fluent responses, but to build a support agent that can:

* Understand the customer's intent.
* Ground responses in historical resolutions.
* Avoid unsupported answers.
* Identify uncertain or high-risk cases.
* Provide an explainable escalation decision.

## Dataset

**Customer Support on Twitter (TWCS)**
Kaggle: `thoughtvector/customer-support-on-twitter`

The project uses one selected brand and a reproducible subsample of the dataset. The full dataset is not included in the repository.

## Tech Stack

**Python · Pandas · Scikit-learn · Sentence Transformers · FAISS · LLM API**

## Evaluation

The system is evaluated using a hand-labelled **150–250 example golden set**.

### Metrics

* **Intent:** Accuracy, Macro F1, per-intent F1
* **Escalation:** Precision, Recall, F1
* **Responses:** Groundedness, correctness, relevance, helpfulness and tone



## Key Failure Modes

The evaluation focuses on failures such as:

* Ambiguous customer messages
* Missing conversation context
* Rare intents
* Incorrect retrieval
* Unsupported or overconfident responses

## Important Limitation

A strong headline metric does not necessarily mean the agent is production-ready. Classification performance, retrieval quality, response quality, and safe escalation must all be considered together.

## Repository Structure

```text
hiver-ai-support-agent/
├── src/              # Core AI pipeline
├── scripts/          # Dataset preparation
├── evaluation/       # Golden set, baselines & evaluation
├── report/           # Results & failure analysis
├── decision_log.md   # Design decisions
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add the required API key.

## Reproduce Results

```bash
python scripts/prepare_brand_data.py
python evaluation/evaluate.py
```

