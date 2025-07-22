# llm-based-reference-free-rag-eval-metrics-meta-eval

This repository contains the code and data used in the paper:

**Meta-Evaluation of LLM-Based Reference-Free Metrics for RAG**  
Xinyuan Cheng · LMU Munich

## 🧠 Overview

We evaluate the robustness and validity of various **reference-free evaluation metrics** for Retrieval-Augmented Generation (RAG), including:

- **Direct Prompting (DP-Free & DP-Token)**
- **G-Eval**
- **GPTScore**
- **RAGAS**

We use the [WikiEval](https://huggingface.co/datasets/ExplodingGradients/WikiEval) benchmark and assess metrics across dimensions of **faithfulness**, **answer relevance**, and **context relevance**.

## 📁 Project Structure

experiment/
├── scripts/ # Scripts for running metric evaluations
├── input/ # QCA data, prompt variants, gold annotations
├── output/ # Metric outputs (e.g., scores, JSON, CSV)

results_analysis/
├── scripts/ # Scripts for correlation, accuracy, and robustness analysis
├── output/ # Final figures, tables, and statistics used in the paper

markdown
复制
编辑

## 🔧 Setup

This project requires Python 3.10+. Recommended packages include:

- `transformers`
- `accelerate`
- `sentence-transformers`
- `scipy`
- `pandas`
- `openpyxl`
- `matplotlib`

You can install dependencies via:

```bash
pip install -r requirements.txt
🚀 Run Experiments
Navigate to experiment/scripts/ and run any of the metric scripts, for example:

python ragas_ff.py --input_csv ../input/ff.csv --output_csv ../output/ragas_ff_output.csv
📊 Analyze Results
All result analysis scripts are in results_analysis/scripts/, including:

Accuracy comparison across metrics and dimensions

Correlation (Spearman/Kendall) with human preferences

Prompt sensitivity analysis

Confidence-level breakdown for DP-Token

bash
python correlation_analysis.py
python dp_token_confidence_check.py
📄 Citation
If you use this code, please cite:

@misc{cheng2025ragmetrics,
  author = {Xinyuan Cheng},
  title = {Meta-Evaluation of LLM-Based Reference-Free Metrics for RAG},
  year = {2025},
  institution = {LMU Munich}
}
📬 Contact
For questions or feedback, please contact:
📧 chengxinyuan@campus.lmu.de
