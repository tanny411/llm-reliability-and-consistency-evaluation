import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_parquet("hf://datasets/open-llm-leaderboard-old/contents/data/train-00000-of-00001-96886cb34a7bc800.parquet")
df.to_csv("old_open_llm_leaderboard_scores.csv", index=False)

global_df = pd.read_csv("analysis_files/ground_truth_conflict.csv")
models = global_df["model"].unique()

models = [model.replace("--" ,"/") for model in models]

df = df[df["fullname"].isin(models)]
df = df[["fullname", "#Params (B)", "Type", "Average ⬆️"]]

df.to_csv("old_open_llm_leaderboard_scores_our_models.csv", index=False)

print(len(models), len(df))