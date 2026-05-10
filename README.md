# Fine-Tuning and RL for LLMs: Intro to Post-Training

> **Course:** [Fine-Tuning and Reinforcement Learning for LLMs: Intro to Post-Training](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/) — DeepLearning.AI × AMD  
> **Instructor:** Sharon Zhou, VP of AI at AMD  
> **Certificate:** Awarded upon successful completion of all graded labs

![Course Certificate](fine-tuning%20rl%20for%20llm%20intro%20to%20post%20training.png)

---

## Overview

This repository contains all six graded lab assignments from the course, covering the full **post-training lifecycle** for large language models — from supervised fine-tuning and RL-based alignment to constitutional AI, evaluation, debugging, and production monitoring.

Each lab is a hands-on implementation exercise building real, working systems, not just walkthroughs. The work spans reward modeling, custom training loops, error-cluster-driven data synthesis, LLM-as-judge evaluation, and production observability pipelines.

---

## Skills Demonstrated

| Skill Area | Tools & Techniques |
|---|---|
| Supervised Fine-Tuning | HuggingFace `transformers`, `trl` SFTTrainer, `completion_only_loss` |
| Reinforcement Learning | GRPO (Group Relative Policy Optimization), reward shaping, partial credit |
| Evaluation & Debugging | Accuracy pipelines, error clustering, PCA visualization, targeted data synthesis |
| Constitutional AI | Prompt template engineering, LLM-as-judge, solution diversity scoring |
| Safety Evaluation | Llama Guard 3, harm category parsing, false positive / false negative metrics |
| Production Monitoring | Latency percentiles, token usage, alert thresholds, semantic clustering |
| Data Engineering | Tokenization, padded batches, embedding similarity search, dataset curation |
| PyTorch / GPU | Mixed precision (bfloat16), inference mode, AMD GPU (MI300X) |

---

## Labs

### Lab 1 — Inspecting Fine-Tuned vs. Base Model

**File:** `Lab1/M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb`

Compared three stages of the DeepSeek Math 7B training pipeline on the **GSM8K** benchmark (grade-school math word problems).

**What I built:**
- A model inference pipeline using a `ServeLLM` wrapper to load, query, and unload GPU models cleanly
- A numerical answer extractor using regex (GSM8K `#### N` format with last-number fallback)
- A full evaluation loop scoring 30 problems per model, tracking accuracy per prompt
- A Llama Guard 3 safety evaluation pipeline with a custom response parser for structured safety outputs (`safe` / `unsafe\nS1\nS5` format)
- Safety metrics: harmful detection rate, benign acceptance rate, false positive rate, false negative rate

**Key results:**

| Model Stage | GSM8K Accuracy |
|---|---|
| Base Model (raw pretrained) | 36.7% |
| SFT Model (instruction-tuned) | 73.3% |
| RL Model (RLHF-aligned) | **86.7%** |

**Concepts applied:** Post-training pipeline stages, safety-correctness tradeoffs, content moderation with Llama Guard, structured output parsing, regex-based answer extraction.

---

### Lab 2 — Supervised Fine-Tuning

**File:** `Lab2/M2_G1_fine_tune_lab_student.ipynb`

Hands-on SFT of **DeepSeek Math 7B Base** on GSM8K using HuggingFace `transformers` and `trl`.

**What I built:**
- Tokenization pipeline: inspected token IDs, token strings, attention masks, and the BOS token
- Embedding analysis: retrieved vocab size (102,400 tokens) and embedding dimensionality (4,096), surfacing that the embedding layer alone holds **419M learnable parameters**
- Padded batch construction for variable-length inputs using `tokenizer(..., padding=True, return_tensors='pt')`
- Fine-tuning runs with HuggingFace `SFTTrainer` using `completion_only_loss=True` to supervise only on answers, not questions
- Systematic learning rate search by reading training and validation loss curves

**Key findings:**

| Learning Rate | Behavior |
|---|---|
| `9e-7` (too low) | Loss barely descends; model fails to learn |
| `1e-5` (optimal) | Training and validation loss both converge cleanly |
| `9e-5` (too high) | Loss diverges or oscillates; model destabilized |

**Concepts applied:** Tokenization internals, embedding spaces, SFT with causal LM loss masking, learning rate schedule effects, loss curve diagnosis.

---

### Lab 3 — GRPO Fine-Tuning

**File:** `Lab3/M2_G2_grpo_finetune_lab_student.ipynb`

Implemented **Group Relative Policy Optimization (GRPO)** to improve a math reasoning model via RL — without requiring a separate critic model.

**What I built:**
- **Answer extractor** — robust regex pipeline to pull numerical answers from verbose model responses
- **Quality analyzer** — detects presence of step-by-step reasoning, intermediate calculations, LaTeX-style expressions, and explicit final-answer markers
- **Reward function with three tiers:**
  - *Unparseable responses* — partial reward based on quality indicators (not a flat 0.0), so the model learns from partial work
  - *Correct answers* — base reward of 1.0 with bonuses for showing clear reasoning steps
  - *Wrong answers* — graded partial credit based on proximity to the correct answer and quality of reasoning shown
- **Evaluation callback** — tracks accuracy and average reward at configurable intervals during training
- **GRPOTrainer configuration** — 12 generations per prompt, group-normalized reward signals, gradient clipping

**Concepts applied:** Policy gradient methods, reward shaping, partial credit design, RL training stability, on-policy generation, group-relative normalization.

---

### Lab 4 — Evaluation and Debugging

**File:** `Lab4/M3_G1_evaluation_and_debugging.ipynb`

Built a full **diagnose → fix → re-evaluate** pipeline for a math reasoning model.

**What I built:**
- Batch evaluation pipeline for 300 GSM8K test problems using GPU-accelerated batch inference
- Error categorization using a local **Llama 3.2 8B** model as an LLM judge, classifying mistakes into: `calculation_error`, `reasoning_error`, `incomplete_solution`, `format_error`, `other`
- Embedding-based similarity search (sentence-transformers `all-MiniLM-L6-v2` + cosine similarity) to select training examples that match each error category
- Targeted fine-tuning on the selected examples with `SFTTrainer`
- Before/after accuracy comparison with full breakdown of newly-correct vs. newly-wrong predictions

**Key results:**

| Stage | Accuracy | Correct / 300 |
|---|---|---|
| Original model | 13.0% | 39 |
| After targeted fine-tuning | **98.7%** | 296 |
| **Improvement** | **+85.7 pp (+659%)** | +257 |

**Error distribution found (261 failures):**

| Error Type | Count | % |
|---|---|---|
| Calculation error | 232 | 88.9% |
| Reasoning error | 21 | 8.0% |
| Incomplete solution | 4 | 1.5% |
| Other / Format | 4 | 1.5% |

**Concepts applied:** Failure-driven data synthesis, embedding similarity retrieval, LLM-as-judge error classification, error clustering with PCA visualization, K-means clustering, the evaluation flywheel.

---

### Lab 5 — Constitutional AI for Mathematical Reasoning

**File:** `Lab5/M4_G1_mathreasoning_student.ipynb`

Applied **Constitutional AI** principles to generate and rank diverse mathematical solutions using **Meta-Llama 3.2 8B Instruct**.

**What I built:**
- **Three distinct prompt templates:**
  - *Chain-of-Thought (CoT)* — step-by-step reasoning from scratch
  - *Verification* — solve + explicitly check the answer makes sense
  - *Alternative method* — given a CoT solution, produce a genuinely different approach (e.g., working backwards, visual/proportional reasoning)
- **Constitutional evaluator** scoring each solution on four principles:
  - **Accuracy** — regex-based extraction + numerical comparison against ground truth (±0.01 tolerance)
  - **Completeness** — regex scan for step indicators, intermediate calculations, and explanatory phrases (0–1 score)
  - **Verification quality** — LLM-as-judge prompt asking whether the solution checks its own answer
  - **Novelty** — LLM-as-judge prompt comparing alternative vs. CoT to verify a genuinely different method
- Weighted composite scoring per solution type and ranking across all three approaches
- Evaluated all 50 problems, saving structured results as JSON

**Concepts applied:** Constitutional AI, solution diversity, LLM-as-judge patterns, prompt template engineering, multi-criteria scoring, answer extraction robustness.

---

### Lab 6 — Production Monitoring and Optimization

**File:** `Lab6/M5_G1_Analysis_and_Optimization_in_Production_student.ipynb`

Simulated a production customer-service LLM deployment and built a full **monitoring, diagnosis, and intervention** pipeline on 120 real production logs.

**What I built:**
- **Monitoring metrics pipeline** computing:
  - Average latency and P95 latency (ms)
  - Average token generation and distribution
  - Error rate percentage (with type breakdown)
  - Average user satisfaction score
- **Alert system** — configurable threshold checker returning human-readable alerts for SLO violations (latency, error rate, satisfaction, verbosity)
- **Failure case extractor** — identifies failures via satisfaction ≤ 2, thumbs-down (-1), or system-detected critical errors
- **Heuristic issue categorizer** — classifies failures into: `too_verbose`, `too_polite`, `knowledge_outdated`, `json_malformed`
- **Semantic clustering** — embeds user prompts + responses with `all-MiniLM-L6-v2`, selects optimal K, and cross-tabulates semantic clusters against heuristic categories
- **Intervention mapping** — matched each failure type to the correct optimization technique: RAG for knowledge freshness, schema guardrails for JSON errors, RL preference tuning for verbosity, prompt/template changes for tone

**Production metrics observed:**

| Metric | Value |
|---|---|
| Average latency | 1,790 ms |
| P95 latency | 3,400 ms |
| Average tokens generated | 876 |
| Error rate | 22.5% |
| Average user satisfaction | 3.26 / 5 |
| Failure cases identified | 56 / 120 (46.7%) |

**Concepts applied:** SLO monitoring, percentile latency, semantic clustering, root-cause diagnosis, RAG, guardrails, RLHF preference tuning — selecting the right tool for the right problem.

---

## Tech Stack

| Area | Details |
|---|---|
| **Models** | DeepSeek Math 7B (Base, SFT, RL), Meta-Llama 3.2 8B Instruct, Llama Guard 3 8B |
| **Fine-tuning** | HuggingFace `transformers`, `trl` (SFTTrainer, GRPOTrainer) |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Dataset** | GSM8K (8,500 grade-school math problems) |
| **Framework** | PyTorch (bfloat16, inference mode), AMD MI300X GPU |
| **Evaluation** | Custom metrics, Constitutional AI, regex answer extraction, LLM-as-judge |
| **Clustering** | K-means, PCA, cosine similarity, scikit-learn |
| **Monitoring** | pandas, numpy percentiles, matplotlib, ipywidgets |

---

## Repository Structure

```
.
├── Lab1/   # Base vs SFT vs RL model comparison + safety evaluation
├── Lab2/   # Supervised fine-tuning with SFTTrainer
├── Lab3/   # GRPO reinforcement learning with custom reward functions
├── Lab4/   # Evaluation, error analysis, and targeted fine-tuning
├── Lab5/   # Constitutional AI solution generation and ranking
└── Lab6/   # Production monitoring, diagnosis, and intervention mapping
```
