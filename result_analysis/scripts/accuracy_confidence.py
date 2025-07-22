import pandas as pd
import ast

high_thresh = 0.9
low_thresh = 0.5

for dim in ["ar", "ff", "cr"]:
    df = pd.read_csv(f"../../experiment/output/{dim}_prompt_variants.csv")
    map_dim_to_sheet = {
        "ar": "Answer Relevance",
        "ff": "Faithfulness",
        "cr": "Context Relevance"
    }
    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=map_dim_to_sheet[dim])

    num_variants_per_sample = 6  # 2 modes × 3 score_ranges
    total_rows = len(df)
    num_samples = total_rows // (num_variants_per_sample * 2)

    high_conf_total, high_conf_correct = 0, 0
    low_conf_total, low_conf_correct = 0, 0

    for i in range(num_samples):
        start_a = i * num_variants_per_sample
        start_b = (i + num_samples) * num_variants_per_sample

        for j in range(num_variants_per_sample):
            idx_a = start_a + j
            idx_b = start_b + j

            row_a = df.iloc[idx_a]
            row_b = df.iloc[idx_b]

            try:
                probs_a = ast.literal_eval(row_a["top_token_probs"])
                probs_b = ast.literal_eval(row_b["top_token_probs"])
                conf_a = max(probs_a.values()) if probs_a else 0
                conf_b = max(probs_b.values()) if probs_b else 0
            except Exception as e:
                print(f"[ERROR parsing top_token_probs] Sample {i}, variant {j}: {e}")
                continue

            try:
                score_a = float(row_a["top1_score"])
                score_b = float(row_b["top1_score"])
            except Exception as e:
                print(f"[ERROR parsing top1_score] Sample {i}, variant {j}: {e}")
                continue

            try:
                model_choice = 1 if score_a > score_b else 2
                human_label = final_df.iloc[i]["Final_answer"]  # 只用 i，不是 j！
                is_correct = int(model_choice == human_label)
            except Exception as e:
                print(f"[ERROR accessing annotation or matching] Sample {i}: {e}")
                continue

            if conf_a > high_thresh and conf_b > high_thresh:
                high_conf_total += 1
                high_conf_correct += is_correct
            elif conf_a < low_thresh and conf_b < low_thresh:
                low_conf_total += 1
                low_conf_correct += is_correct

    # 输出
    print(f"\n=== DIM: {dim} ===")
    if high_conf_total > 0:
        acc = high_conf_correct / high_conf_total * 100
        print(f"High Confidence: Accuracy = {acc:.2f}% ({high_conf_correct}/{high_conf_total})")
    else:
        print("High Confidence: No samples")

    if low_conf_total > 0:
        acc = low_conf_correct / low_conf_total * 100
        print(f"Low Confidence: Accuracy = {acc:.2f}% ({low_conf_correct}/{low_conf_total})")
    else:
        print("Low Confidence: No samples")
