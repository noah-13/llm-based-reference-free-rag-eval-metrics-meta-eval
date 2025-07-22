# Meta-Evaluation of Reference-Free LLM Metrics for RAG

This repository contains the code and data used in the paper:

**Meta-Evaluation of LLM-Based Reference-Free Metrics for RAG**  
Xinyuan Cheng · LMU Munich

## 🧠 Overview

We evaluate the robustness and validity of various **reference-free evaluation metrics** for Retrieval-Augmented Generation (RAG), including:

- **Direct Prompting** (DP-Free & DP-Token)
- **G-Eval**
- **GPTScore**
- **RAGAS**

Our evaluation is conducted on the [WikiEval benchmark](https://huggingface.co/datasets/ExplodingGradients/WikiEval) and covers three quality dimensions:

- **Faithfulness**
- **Answer Relevance**
- **Context Relevance**

We assess both **pairwise accuracy** and **correlation with human judgments**, and analyze the effects of prompt format and LLM confidence.

## 📁 Project Structure

<pre> ```text project-name/ ├── folder1/ │ ├── file1.py │ └── file2.py ├── folder2/ │ └── subfolder/ │ └── file3.py └── README.md ``` </pre>
