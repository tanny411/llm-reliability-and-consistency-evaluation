import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *


# combines responses from a single model, and single prompt type (e.g 2 options)
def combine_responses(
    path,
    file_template="classification_response_P{}.csv",
    old_cols=None,
    full_text=False,
):
    common_cols = [
        "index",
        "text",
        "tags",
        "sub_tags",
        "ground_truth",
        "paper link",
        "data_source",
    ]
    different_columns = ["text_response", "prompt", "text_w_input", "logprobs"]

    if old_cols:
        common_cols = old_cols
        different_columns = ["text_response", "prompt"]

    horizontally_combined_df = None

    vertical_cols = common_cols + different_columns
    # if not full_text:
    #     vertical_cols += ["text_response_prob"]

    vertically_combined_df = pd.DataFrame([], columns=vertical_cols)

    num_prompts = len(prompt_questions)
    if old_cols:
        num_prompts = 4

    for i in range(num_prompts):
        filename = path + file_template.format(i)

        df = pd.read_csv(filename, converters={"text_response": str})

        horizontal_columns = different_columns
        # if not full_text:
        #     horizontal_columns += ["text_response_prob"]

        modified_horizontal_columns = [col + f"_P{i}" for col in horizontal_columns]

        for new_col, col in zip(modified_horizontal_columns, horizontal_columns):
            df[new_col] = df[col]

        # combine horizontaly
        if horizontally_combined_df is None:
            horizontally_combined_df = df[common_cols + modified_horizontal_columns]
        else:
            horizontally_combined_df = horizontally_combined_df.merge(
                df[["index"] + modified_horizontal_columns], on="index"
            )

        # combine vertically
        df["prompt"] = f"P{i}"
        vertically_combined_df = pd.concat(
            [vertically_combined_df, df[vertical_cols]], ignore_index=True
        )

    return horizontally_combined_df, vertically_combined_df


def yes_no_finder(text):
    # returns yes, no, or neither
    if text.lower().startswith("yes"):
        return "Yes"
    if text.lower().startswith("no"):
        return "No"

    return "Neither"


def analysis(main_directory):
    # needs all prompts responses to be collected before running analysis

    palette = {
        "YES": "tab:green",
        "NO": "tab:red",
        "Yes": "tab:green",
        "No": "tab:red",
        "Neither": "tab:blue",
        "A": "tab:green",
        "B": "tab:red",
        "C": "tab:blue",
        "D": "tab:gray",
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))

    hue_order = ["YES", "NO"]
    _, vertically_combined_2_class_df = combine_responses(main_directory + "2_options/")
    sns.countplot(
        ax=axes[0],
        data=vertically_combined_2_class_df,
        x="prompt",
        hue="text_response",
        palette=palette,
        hue_order=hue_order,
    ).set_title("YES,NO / 2 options prompt LOL")

    hue_order = ["Yes", "No", "Neither"]
    _, vertically_combined_3_class_df = combine_responses(main_directory + "3_options/")
    sns.countplot(
        ax=axes[1],
        data=vertically_combined_3_class_df,
        x="prompt",
        hue="text_response",
        palette=palette,
        hue_order=hue_order,
    ).set_title("Yes,No,Neither / 3 options prompt")

    hue_order = ["A", "B", "C", "D"]
    _, vertically_combined_4_class_df = combine_responses(main_directory + "4_options/")
    sns.countplot(
        ax=axes[2],
        data=vertically_combined_4_class_df,
        x="prompt",
        hue="text_response",
        palette=palette,
        hue_order=hue_order,
    ).set_title("MCQ / 4 options prompt")

    hue_order = ["A", "B", "C", "D"]
    _, vertically_combined_4_class_randomized_df = combine_responses(
        main_directory + "4_options/randomized/"
    )
    sns.countplot(
        ax=axes[3],
        data=vertically_combined_4_class_randomized_df,
        x="prompt",
        hue="text_response",
        palette=palette,
        hue_order=hue_order,
    ).set_title("MCQ / 4 options prompt randomized")

    hue_order = ["Yes", "No", "Neither"]
    _, vertically_combined_full_text_df = combine_responses(
        main_directory + "full_text/",
        file_template="full_text_response_P{}.csv",
        full_text=True,
    )
    temp_df = vertically_combined_full_text_df.copy(deep=True)
    temp_df["text_response"] = temp_df["text_response"].apply(yes_no_finder)
    sns.countplot(
        ax=axes[4],
        data=temp_df,
        x="prompt",
        hue="text_response",
        palette=palette,
        hue_order=hue_order,
    ).set_title("Yes,No,Neither / full text")

    for ax in axes:
        ax.set_ylim(0, 900)

    output_directory = main_directory + "figs/"
    check_and_create_folder(output_directory)

    fig.tight_layout()
    fig.savefig(output_directory + "all_prompt_responses.png")


def dataset_analysis(data_path):
    df = pd.read_csv(data_path)

    print("Size of dataset:", len(df))
    print("Dataset columns:", df.columns.tolist())
    print("Dataset categories:", df["tags"].unique())
    print()
    print(df["tags"].value_counts())
    print()
    print(df.groupby(["tags", "ground_truth"]).count()["index"])
    print()
    print(df["ground_truth"].value_counts())


# dataset_analysis("../../../curated_datasets/version_3/data_v3.csv")


# combines responses of all models, all prompts, all cls prompt types (2/3/4 options)
def combine_all_responses_cls(models, directory):
    cls_response_types = [
        "2_options",
        "3_options",
        "4_options",
        "4_options/randomized",
        "4_options/option_probs",
        "4_options/randomized/option_probs",
    ]
    cls_response_data = pd.DataFrame([])
    for model in models:
        for response_type in cls_response_types:
            for prompt in range(5):
                file_suffix = f"classification_response_P{prompt}.csv"
                filename = "/".join([directory, model, response_type, file_suffix])
                try:
                    df = pd.read_csv(filename)
                except Exception as e:
                    # file not found: no need to add it to combined df
                    print(e)
                    continue
                
                col_list = ["index", "text", "tags", "sub_tags", "ground_truth", "paper link", "data_source"]
                if "text_response_prob" in df.columns:
                    col_list += ["text_response_prob"]
                if "total_prob" in df.columns:
                    col_list += ["total_prob"]
                if "normalized_text_response_prob" in df.columns:
                    col_list += ["normalized_text_response_prob"]
                
                new_df = df[col_list]

                if "new_response" in df.columns:
                    new_df["text_response"] = df["new_response"]
                elif "response_trimmed" in df.columns: # for non-mcq responses
                    new_df["text_response"] = df["response_trimmed"]
                elif "text_response" in df.columns: # for openai models
                    new_df["text_response"] = df["text_response"]
                else:
                    print("Error! Can't find response columns!")

                if "response" in df.columns:
                    new_df["response"] = df["response"]
                elif "original_text_response" in df.columns:
                    new_df["response"] = df["original_text_response"]
                elif "text_response" in df.columns:
                    new_df["response"] = df["text_response"]
                else:
                    print("Error! Can't find original response columns!")
                
                
                new_df["model"] = model
                new_df["response_type"] = response_type
                new_df["prompt_type"] = prompt

                
                cls_response_data = pd.concat(
                    [cls_response_data, new_df], ignore_index=True
                )
        print(len(cls_response_data))

    return cls_response_data


def combine_all_responses_full_text(models, directory):
    response_data = pd.DataFrame([])
    for model in models:
        response_type = "full_text"
        for prompt in range(5):
            file_suffix = f"full_text_response_P{prompt}.csv"
            filename = "/".join([directory, model, response_type, file_suffix])
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                # file not found: no need to add it to combined df
                print(e)
                continue
            df["model"] = model
            df["response_type"] = response_type
            df["prompt_type"] = prompt
            response_data = pd.concat([response_data, df], ignore_index=True)
        print(len(response_data))

    return response_data


# gets list of folders with model names
def get_model_list(directory):
    model_list = [
        name
        for name in os.listdir(directory)
        if not name.startswith("old")
        and not name.startswith("test")
        and not "model_response" in name
        and not "partials" in name
        and os.path.isdir(os.path.join(directory, name))
    ]
    return model_list


def combine_all_responses(directory):
    model_list = get_model_list(directory)
    
    cls_response_data = combine_all_responses_cls(model_list, directory)
    cls_response_data.to_csv(directory + "/all_model_responses_cls.csv", index=False)

    # response_data = combine_all_responses_full_text(model_list, directory)
    # response_data.to_csv(directory + "/all_model_responses_full_text.csv", index=False)


# combine_all_responses("model_responses")
