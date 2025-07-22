import argparse
import json
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util


def safe_parse_json_list_with_error(output: str) -> tuple[list[str] | None, str]:
    def attempt_parse(candidate: str):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed, "ok"
            return None, "not_a_list"
        except json.JSONDecodeError:
            return None, "json_parse_error"

    parsed, status = attempt_parse(output)
    if status.startswith("ok"):
        return parsed, status

    for pattern in [r'(\[\s*.*?\])']:
        matches = re.finditer(pattern, output, re.DOTALL)
        for match in matches:
            candidate = match.group(1)

            stack = []
            for i, ch in enumerate(candidate):
                if ch == "[":
                    stack.append("]")
                elif ch == "]" and stack:
                    stack.pop()
                if not stack:
                    candidate = candidate[:i + 1]
                    break

            parsed, status = attempt_parse(candidate)
            if status.startswith("ok"):
                return parsed, "extracted_middle"

    match = re.search(r'\[\s*.*', output, re.DOTALL)
    if match:
        candidate = match.group(0)
        if candidate.count('[') > candidate.count(']'):
            candidate += ']'
        candidate = re.sub(r',\s*]', ']', candidate)

        parsed, status = attempt_parse(candidate)
        if status.startswith("ok"):
            return parsed, "partial_fix_list"

        string_items = re.findall(r'"([^"]*)"', candidate)
        if string_items:
            return string_items, "partial_fix_list_truncated"

    return None, "parse_failed"


class ContextRelevanceEvaluator:
    def __init__(self, model_id: str, embed_model=None, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.model.eval()

        self.embed_model = embed_model
        self.device = device

        self.spacy_nlp = English()
        self.spacy_nlp.add_pipe("sentencizer", config={"overwrite": True})

    def spacy_sent_tokenize(self, text: str) -> list[str]:
        doc = self.spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _generate(self, prompt: str, max_new_tokens=512):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                do_sample=False
            )

        generated_len = outputs.shape[-1] - input_ids.shape[-1]
        truncated = generated_len >= max_new_tokens
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), truncated


    def build_context_relevance_prompt(self, question: str, context: str) -> str:
        instruction = "Given a question and a context, extract only the sentences from the context that help answer the question."

        prompt = f"""{instruction}

        The extracted sentences must:
        - Be copied exactly from the original context.
        - Be semantically aligned with the statements.
        - Be returned as a JSON list of strings.
        - Do not rewrite or change any sentence.
        - If no relevant sentence is found, return exactly: ["Insufficient Information"]

        Now consider the following:

        Question:{question}
        Context:
        {context}
        
        Output:"""

        return prompt



    def evaluate(self, question: str, context: str, cutoff=0.5) -> dict:
        sentence_list = self.spacy_sent_tokenize(context)

        result = {
            "context_relevance_score": np.nan,
            "total_sentences": len(sentence_list),
            "extracted_sentences": [],
            "raw_output": "",
            "parse_status": "",
            "statements_truncated": 0
        }

        if not sentence_list or question.strip() == "":
            result["context_relevance_score"] = 0.0
            return result
        
        prompt = self.build_context_relevance_prompt(question, context)
        output, truncated = self._generate(prompt)        
        result["raw_output"] = output
        result["statements_truncated"] = int(truncated)

        parsed_list, status = safe_parse_json_list_with_error(output)
        result["parse_status"] = status

        if parsed_list is None or parsed_list == ["Insufficient Information"]:
            result["context_relevance_score"] = 0.0
            return result

        try:
            question_emb = self.embed_model.encode(question, convert_to_tensor=True, device=self.device)
            parsed_embs = self.embed_model.encode(parsed_list, convert_to_tensor=True, device=self.device)
            sim_scores = util.cos_sim(question_emb, parsed_embs)[0]
            selected = [parsed_list[i] for i in range(len(parsed_list)) if sim_scores[i] >= cutoff]

            ctx_embs = self.embed_model.encode(sentence_list, convert_to_tensor=True, device=self.device)
            extracted = []
            for s in selected:
                s_emb = self.embed_model.encode(s, convert_to_tensor=True, device=self.device)
                cos_sim = util.cos_sim(s_emb, ctx_embs)[0]
                best_idx = int(torch.argmax(cos_sim).item())
                extracted.append(sentence_list[best_idx])
        except Exception as e:
            print(f"[Embedding error] {e}")
            extracted = []

        result["extracted_sentences"] = extracted
        result["context_relevance_score"] = len(extracted) / len(sentence_list) if sentence_list else 0.0
        return result


def main(args):
    df = pd.read_csv(args.input_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    for cutoff in args.cutoffs:
        all_records = []


        evaluator = ContextRelevanceEvaluator(
            model_id=args.model_id,
            embed_model=embed_model,
            device=device
        )

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"cutoff={cutoff:.2f}"):
            try:
                q = str(row["question"]).strip()
                ctx = str(row["context"]).strip()
                result = evaluator.evaluate(q, ctx, cutoff=cutoff)
            except Exception as e:
                print(f"[Error] {e}")
                result = {
                    "context_relevance_score": np.nan,
                    "total_sentences": 0,
                    "extracted_sentences": [],
                    "raw_output": "ERROR",
                    "parse_status": "exception",
                    "statements_truncated": 0
                }

            record = row.to_dict()
            record.update({
                "context_relevance_score": result["context_relevance_score"],
                "total_sentences": result["total_sentences"],
                "extracted_sentences": json.dumps(result["extracted_sentences"], ensure_ascii=False),
                "raw_output": result["raw_output"],
                "parse_status": result["parse_status"],
                "statements_truncated": result["statements_truncated"],
            })
            all_records.append(record)

        df_out = pd.DataFrame(all_records)
        suffix = str(int(cutoff * 100)).zfill(3)
        output_file = args.output_csv
        df_out.to_csv(output_file, index=False)
        print(f"[✓] Saved merged results to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--cutoffs", type=float, nargs="+",
        default=[0.5], help="List of cutoffs to try"
    )
    args = parser.parse_args()
    main(args)
