import pandas as pd

def combine_all():
    small = pd.read_csv("model_responses/small_model_runs/processed_model_responses_cls/all_model_responses_cls.csv", index_col=0)
    small["model_type"] = "small"

    large = pd.read_csv("model_responses/large_model_runs/processed_model_responses_cls/all_model_responses_cls.csv", index_col=0)
    large["model_type"] = "large"

    openai = pd.read_csv("model_responses/openai/all_model_responses_cls.csv", index_col=0)
    openai["model_type"] = "openai"

    full_df = pd.concat([small, large, openai])
    full_df["model"] = full_df["model"].apply(lambda x: x.split("--")[-1])
    full_df.to_csv("model_responses/full_all_model_responses_cls.csv")

    ## sanity check
    print("Number of rows:", len(full_df), len(small)+len(large)+len(openai))
    print("Number of models:", full_df["model"].nunique()) # 26+5+4


combine_all()