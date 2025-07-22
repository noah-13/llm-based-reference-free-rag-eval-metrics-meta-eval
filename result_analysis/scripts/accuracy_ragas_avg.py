import pandas as pd
import numpy as np

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

# 收集结果
final_results = {}

for dim in ["ar", "ff", "cr"]:
    accuracies = []  # 保存所有 mode 的准确率
    for mode in ["", "chat_"]:
        df = pd.read_csv(f"../../experiment/output/{dim}_ragas_{mode}output.csv")
        final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=map_dim_to_sheet[dim])
        df["pair_id"] = df.index % 50

        correct = 0
        total = 0

        for i in range(50):
            row_a = df[df.index == i].iloc[0]
            row_b = df[df.index == i + 50].iloc[0]

            score_a = row_a[f"{map_dim_to_score[dim]}_score"]
            score_b = row_b[f"{map_dim_to_score[dim]}_score"]

            model_choice = 1 if score_a > score_b else 2
            human_label = final_df.iloc[i]["Final_answer"]

            if model_choice == human_label:
                correct += 1
            total += 1

        accuracy = correct / total
        accuracies.append(accuracy)

        print(f"Dim: {map_dim_to_sheet[dim]} | Mode: {mode or 'plain'} | Accuracy: {accuracy * 100:.2f}%")

    # 计算平均和标准差
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    final_results[map_dim_to_sheet[dim]] = (mean_acc, std_acc)

# 打印结果用于主文表格
print("\nSummary (mean ± std):")
for dim, (mean, std) in final_results.items():
    print(f"{dim}: {mean*100:.2f}% ± {std*100:.2f}%")
