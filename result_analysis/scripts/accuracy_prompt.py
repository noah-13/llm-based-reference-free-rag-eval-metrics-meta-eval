import pandas as pd

for dim in ["ar", "ff","cr"]:
    df = pd.read_csv(f"../../experiment/output/{dim}_prompt_variants.csv")
    map_dim_to_sheet = {
        "ar": "Answer Relevance",
        "ff": "Faithfulness",
        "cr": "Context Relevance"
    }
    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=map_dim_to_sheet[dim])

    score_columns = ["explicit_score", "weighted_score", "top1_score"]
    modes = ["imperative", "interrogative"]

    results = {}
    table_rows = []  

    for score_col in score_columns:
        results[score_col] = {}

        for mode in modes:
            sub_df = df[df["mode"] == mode].sort_index().reset_index(drop=True)

            if len(sub_df) % 3 != 0:
                print(f"[WARNING] Row count for mode={mode} is not a multiple of 3 ({len(sub_df)} rows)")

            num_samples = len(sub_df) // 3 // 2  
            results[score_col][mode] = {}

            for i in range(num_samples):
                start_a = i * 3
                start_b = (i + num_samples) * 3

                rows_a = sub_df.iloc[start_a:start_a + 3]
                rows_b = sub_df.iloc[start_b:start_b + 3]

                for idx in range(3):  # 3 variants per sample
                    row_a = rows_a.iloc[idx]
                    row_b = rows_b.iloc[idx]
                    range_str = row_a["score_range"]

                    if row_b["score_range"] != range_str:
                        print(f"[ERROR] score_range mismatch at sample={i}, mode={mode}, variant={idx}")
                        continue

                    try:
                        low, high = map(float, range_str.split("-"))
                        score_a = float(row_a[score_col])
                        score_b = float(row_b[score_col])
                    except Exception as e:
                        print(f"[ERROR] Failed to parse score/sample at i={i}, col={score_col}: {e}")
                        continue

                    if range_str not in results[score_col][mode]:
                        results[score_col][mode][range_str] = {
                            "correct": 0,
                            "total": 0,
                            "out_of_range": 0,
                        }

                    out_of_range = not (low <= score_a <= high) or not (low <= score_b <= high)
                    if out_of_range:
                        results[score_col][mode][range_str]["out_of_range"] += 1

                    model_choice = 1 if score_a > score_b else 2
                    human_label = final_df.iloc[i]["Final_answer"]

                    if model_choice == human_label:
                        results[score_col][mode][range_str]["correct"] += 1
                    results[score_col][mode][range_str]["total"] += 1

    for score_col in score_columns:
        for mode in modes:
            for range_str, stats in results[score_col][mode].items():
                acc = stats["correct"] / stats["total"] * 100
                oor = stats["out_of_range"] / stats["total"] * 100
                table_rows.append({
                    "score_type": score_col,
                    "mode": mode,
                    "score_range": range_str,
                    "accuracy (%)": round(acc, 2),
                    "out_of_range (%)": round(oor, 2),
                    "count": stats["total"],
                })

    result_table = pd.DataFrame(table_rows)

    result_table.to_csv(f"{dim}_prompt_results_summary.csv", index=False)

    print("\n=== Summary Table ===")
    print(result_table)
