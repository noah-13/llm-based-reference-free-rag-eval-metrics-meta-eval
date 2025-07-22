import pandas as pd
import ast  # To safely evaluate string representations of lists

for dim in ["ar", "ff","cr"]:
    for mode in ["", "chat_"]:
        df = pd.read_csv(f"../../experiment/output/{dim}_ragas_{mode}output.csv")
        map_dim_to_sheet = {
            "ar": "Answer Relevance",
            "ff": "Faithfulness",
            "cr": "Context Relevance"
        }
        map_dim_to_score = {
            "ar": "answer_relevance",
            "ff": "faithfulness",
            "cr": "context_relevance"
        }
        final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=map_dim_to_sheet[dim])
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

            score_a = row_a[f"{map_dim_to_score[dim]}_score"]
            score_b = row_b[f"{map_dim_to_score[dim]}_score"]

            model_choice = 1 if score_a > score_b else 2
            human_label = final_df.iloc[i]["Final_answer"]

            if model_choice == human_label:
                correct += 1
            total += 1

        accuracy = correct / total

        print(f"Dim:{map_dim_to_sheet[dim]} Mode: {mode} Accuracy: {accuracy * 100:.2f}%")
