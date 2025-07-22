import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from itertools import product
from tqdm import tqdm
import argparse

def extract_explicit_score_from_text(text: str, score_range=(1.0, 5.0)) -> float | None:
    pattern = r"\b\d+(?:\.\d+)?\b"
    matches = re.findall(pattern, text)
    for m in matches:
        try:
            val = float(m)
            if score_range[0] <= val <= score_range[1]:
                return val
        except ValueError:
            continue
    return None


def extract_weighted_score_llama(first_logits, tokenizer, score_range=(1, 5)):
    probs = F.softmax(first_logits, dim=-1)
    weighted_score, total_prob = 0.0, 0.0
    for i in range(score_range[0], score_range[1] + 1):
        token_ids = tokenizer(str(i), add_special_tokens=False)["input_ids"]
        if len(token_ids) == 1:
            token_id = token_ids[0]
            prob = probs[token_id].item()
            weighted_score += prob * i
            total_prob += prob
    return weighted_score / total_prob if total_prob > 0 else None

def get_top_token_probs_llama(first_logits, tokenizer, score_range=(1, 5)):
    probs = F.softmax(first_logits, dim=-1)
    raw_probs = {}
    for i in range(score_range[0], score_range[1] + 1):
        token_ids = tokenizer(str(i), add_special_tokens=False)["input_ids"]
        if len(token_ids) == 1:
            token_id = token_ids[0]
            raw_probs[str(i)] = probs[token_id].item()
    total = sum(raw_probs.values())
    if total == 0:
        return {}
    return {
        k: round(v / total, 4)
        for k, v in sorted(raw_probs.items(), key=lambda x: x[1], reverse=True)
    }

def build_prompt_variant(context, question, mode="imperative", score_range=(1, 5)):
    score_str = f"{score_range[0]} (irrelevant) to {score_range[1]} (relevant)"
    instruction = {
        "imperative": f"Rate the relevance of the context from {score_str}.",
        "interrogative": f"How relevant is the context to the question on a scale from {score_str}?"
    }.get(mode)
    if instruction is None:
        raise ValueError("mode must be 'imperative' or 'interrogative'")
    return f"""{instruction}
    A good context contains only the necessary information without unrelated or noisy details. Extra information that distracts or dilutes the relevance makes the context less useful.

    Question: {question}
    Context: {context}
    Score ({score_range[0]}-{score_range[1]}):"""

def evaluate_llama3_faithfulness(model, tokenizer, prompt, score_range=(1, 5)):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id and eot_id not in eos_ids:
            eos_ids.append(eot_id)
    except:
        pass
    output = model.generate(
        input_ids,
        max_new_tokens=8,
        eos_token_id=eos_ids,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False
    )
    gen_tokens = output.sequences[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    first_logits = output.scores[0].squeeze(0)
    weighted_score = extract_weighted_score_llama(first_logits, tokenizer, score_range)
    top_probs = get_top_token_probs_llama(first_logits, tokenizer, score_range)
    top1_token = next(iter(top_probs)) if top_probs else None
    top1_score = int(top1_token) if top1_token and top1_token.isdigit() else None
    explicit_score = extract_explicit_score_from_text(gen_text, score_range)
    return {
        "prompt": prompt,
        "generated_text": gen_text,
        "explicit_score": explicit_score,
        "weighted_score": round(weighted_score, 3) if weighted_score is not None else None,
        "top_token_probs": top_probs,
        "top1_token": top1_token,
        "top1_score": top1_score
    }

def main(input_csv, output_csv, model_id):
    df = pd.read_csv(input_csv)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16).eval()

    modes = ["imperative", "interrogative"]
    ranges = [(1, 5), (1, 10), (0, 1)]

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        c, q = row["context"], row["question"]
        for mode, score_range in product(modes, ranges):
            prompt = build_prompt_variant(c, q, mode, score_range)
            eval_result = evaluate_llama3_faithfulness(model, tokenizer, prompt, score_range)
            eval_result.update({
                "context": c,
                "question": q,
                "mode": mode,
                "score_range": f"{score_range[0]}-{score_range[1]}"
            })
            results.append(eval_result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Done. Results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with context, question")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.model_id)
