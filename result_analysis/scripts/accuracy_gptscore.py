import pandas as pd

for dim in ["ar", "ff","cr"]:
    df = pd.read_csv(f"../../experiment/output/{dim}_gptscore_variants.csv")

    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name="Answer Relevance")

    df["pair_id"] = df.index % 50

    results = {}

    for mode in ["imperative", "interrogative"]:
        sub_df = df[df["mode"] == mode].sort_index().reset_index(drop=True)

        
        correct = 0
        total = 0
        
        for i in range(50):
            row_a = sub_df[sub_df.index == i].iloc[0]
            row_b = sub_df[sub_df.index == i + 50].iloc[0]

            score_a = row_a["gptscore"]
            score_b = row_b["gptscore"]

            model_choice = 1 if score_a > score_b else 2
            human_label = final_df.iloc[i]["Final_answer"]

            if model_choice == human_label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        results[mode] = accuracy

    print(f"{dim.upper()} - Imperative Accuracy: {results['imperative'] * 100:.2f}%")
    print(f"{dim.upper()} - Interrogative Accuracy: {results['interrogative'] * 100:.2f}%")
