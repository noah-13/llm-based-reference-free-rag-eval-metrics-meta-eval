import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def build_prompt(question, mode="imperative"):
    if mode == "imperative":
        instruction = "Generate a relevant answer to the question. A relevant answer is complete, specific, and directly answers all parts of the question. Answers that are vague, off-topic, or incomplete are less relevant."
    else:
        instruction = "Could you generate a relevant answer to the question? A relevant answer is complete, specific, and directly answers all parts of the question. Answers that are vague, off-topic, or incomplete are less relevant."
        return f"""{instruction}

Question: {question}
Answer:"""


def gptscore(model, tokenizer, question, answer, mode="imperative"):
    prompt = build_prompt(question, mode)
    full_text = prompt + " " + answer

    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    gen_logits = logits[:, prompt_ids.shape[-1] - 1:-1, :]
    gen_ids = full_ids[:, prompt_ids.shape[-1]:]

    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.mean().item()


def main(input_csv, output_csv, model_id):
    df = pd.read_csv(input_csv)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16).eval()

    modes = ["imperative", "interrogative"]
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row.get("question", "Unknown question")
        answer = row["answer"]

        for mode in modes:
            score = gptscore(model, tokenizer, question, answer, mode)
            results.append({
                "question": question,
                "answer": answer,
                "mode": mode,
                "gptscore": score
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Done. Results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with columns: question, answer")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.model_id)
