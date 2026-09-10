# Hiver AI Support Agent — Evaluation Report

## 1. Problem Framing

The goal of this project is to build an AI customer-support agent for a single brand represented in the Customer Support on Twitter dataset.

The system receives an incoming customer message and performs four tasks:

1. Classifies the customer's intent.
2. Retrieves historically similar customer-support interactions.
3. Generates a response grounded in those historical examples.
4. Decides whether the request can be automatically handled or should be escalated to a human.

### What Good Looks Like

A good support agent should:

- correctly understand the customer's problem;
- retrieve relevant historical evidence;
- provide a concise and useful response;
- avoid unsupported claims;
- maintain an appropriate support tone;
- recognize uncertain or high-risk cases;
- escalate cases when automatic handling is unsafe.

### What We Are Not Building

This project is not intended to replace a complete production customer-support platform.

It does not attempt to:

- integrate with real payment systems;
- issue refunds automatically;
- modify customer accounts;
- authenticate customers;
- access private customer information;
- provide a complete frontend;
- guarantee production-level safety.

The focus is the AI decision pipeline and its evaluation.

---

# 2. Dataset and Sampling

The project uses the Customer Support on Twitter dataset (TWCS).

The dataset contains customer-support interactions between users and brand support accounts.

A single support brand was selected to keep the task focused on a consistent support style and resolution behavior.

The full dataset was processed in chunks rather than loaded completely into memory.

Historical customer messages were paired with their corresponding support responses using the response tweet identifiers.

### Sampling

The final evaluation set contains approximately 150–250 manually labelled customer messages.

The golden set was kept separate from the training examples used by the classifier.

The intent labels were defined from recurring problems observed in the selected brand's conversations.

---

# 3. Intent Taxonomy

The final intent taxonomy was deliberately kept small.

Example categories include:

- account_access
- payment_issue
- refund
- duplicate_charge
- delivery
- cancellation
- technical_issue
- subscription
- verification
- other

The final categories were adjusted after inspecting the selected brand's actual conversations.

The goal was to avoid creating overly fine-grained categories that would be difficult to distinguish consistently.

---

# 4. System Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Semantic Retrieval
       |
       v
Historical Support Evidence
       |
       v
LLM Response Generation
       |
       v
Escalation Decision
       |
       +------------------+
       |                  |
       v                  v
 AUTO-HANDLE          ESCALATE
       |                  |
       v                  v
 Generated Reply      Human Review