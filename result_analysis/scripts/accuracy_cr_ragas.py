import pandas as pd
import ast  # To safely evaluate string representations of lists

for mode in ["", "chat_"]:
    df = pd.read_csv(f"../../experiment/output/cr_ragas_{mode}output.csv")

    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name="Context Relevance")

    df["pair_id"] = df.index % 50

    results = {}
    output_parse_status_0 = 0  
    output_parse_status_1 = 0  

    sub_df = df  

    correct = 0
    total = 0

    for i in range(50):
        row_a = sub_df[sub_df.index == i].iloc[0]
        row_b = sub_df[sub_df.index == i + 50].iloc[0]

        score_a = row_a["context_relevance_score"]
        score_b = row_b["context_relevance_score"]

        model_choice = 1 if score_a > score_b else 2
        human_label = final_df.iloc[i]["Final_answer"]

        if model_choice == human_label:
            correct += 1
        total += 1

        try:
            output_parse_status_a = row_a["parse_status"]
            output_parse_status_b = row_b["parse_status"]
            status_list = ['extracted_middle', 'ok', 'partial_fix_list_truncated', 'partial_fix_list']
            if output_parse_status_a in status_list:
                output_parse_status_1 += 1
            if output_parse_status_b in status_list:
                output_parse_status_1 += 1
        except (ValueError, SyntaxError):
            pass

    accuracy = correct / total

    print(f"mode: {mode} Accuracy: {accuracy * 100:.2f}%")

    print(f"Total number of failiure in output_parse_status: {output_parse_status_0}")
    print(f"Total number of success in output_parse_status: {output_parse_status_1}")
