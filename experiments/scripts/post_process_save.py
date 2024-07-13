import os
import ast
import argparse
import numpy as np
import pandas as pd
from warnings import filterwarnings
from utils import check_and_create_folder
from post_process import post_process_all_responses
from api_call_utils import fix_classes, mcq_choices_list
from analysis import get_model_list, combine_all_responses, combine_all_responses_cls

filterwarnings('ignore')

def get_cls_proper_model_list(directory, model_list, include_probs=True):
    # returns list of models that contain all cls responses
    proper_model_list = []

    for model_name in model_list:
        proper = True
        response_types = ["2_options", "3_options", "4_options", "4_options/randomized"]
        if include_probs:
            response_types += ["4_options/option_probs", "4_options/randomized/option_probs"]
        for response_type in response_types:
            for prompt_type in range(5):
                file_suffix = f"classification_response_P{prompt_type}.csv"
                filename = "/".join([directory, model_name, response_type, file_suffix])
                if not os.path.exists(filename):
                    proper = False
                    break
            if not proper:
                break
        if proper:
            proper_model_list.append(model_name)

    return proper_model_list

def get_fully_proper_model_list(directory, proper_model_list):
    # returns list of models that contains all cls AND full-text responses
    fully_proper_model_list = []

    for model_name in proper_model_list:
        proper = True
        response_type = "full_text"
        for prompt_type in range(5):
            file_suffix = f"full_text_response_P{prompt_type}.csv"
            filename = "/".join([directory, model_name, response_type, file_suffix])
            if not os.path.exists(filename):
                proper = False
                break
        if proper:
            fully_proper_model_list.append(model_name)

    return fully_proper_model_list

def post_process_mcq_option_probs(directory, model_name):
    ## Get responses from "option probs MCQ" type
    ## Take form original response and save as processed response
    response_types = ["4_options/option_probs", "4_options/randomized/option_probs"]
    
    for response_type in response_types:
        
        save_dir = "/".join([directory, "processed_model_responses_cls", model_name, response_type])
        check_and_create_folder(save_dir)

        for prompt_type in range(5):
            file_suffix = f"classification_response_P{prompt_type}.csv"
            filename = "/".join([directory, model_name, response_type, file_suffix])
            try: 
                df = pd.read_csv(filename)
            except:
                # print(model_name)
                continue
            
            df["logprobs"] = df["logprobs"].apply(lambda x: ast.literal_eval(x))
            df["response_trimmed"] = df["logprobs"].apply(lambda x: x["tokens"][np.argmax(x["token_logprobs"])]) 
            df["text_response_prob"] = df["logprobs"].apply(lambda x: 100*np.exp(np.max(x["token_logprobs"])))

            ## make normalized probabilities
            df["total_prob"] = df["logprobs"].apply(lambda x: 100*sum([np.exp(prob) for prob in x["token_logprobs"]]))
            df["normalized_text_response_prob"] = 100*df["text_response_prob"]/df["total_prob"]

            df.to_csv("/".join([save_dir, file_suffix]), index=False)

mcq_choices_dict = {key:val for d in mcq_choices_list for key,val in d.items()}

def extract_choices_list(text):
    choices_list = []
    text = text.split("Choices:\n", maxsplit=1)[1]
    text = text.split("\n\n", maxsplit=1)[0]
    text = text.split("\n")
    for line in text:
        letter, option = line.split(".", maxsplit=1)
        option = option.strip()
        original_letter = mcq_choices_dict[option]
        choices_list.append({option:original_letter})
    return choices_list

def calibrate_mcq_responses(directory, model_name):
    ## Calibrate MCQ responses
    # Fix "A__fixed" responses
    # Calibrate randomized mcq to original mcq choices list
    # Large models don't have the entire choices list, need to extract the choices list form the prompt.

    # **Processed files are re-writtern.**

    response_types = ["4_options", "4_options/randomized", "4_options/option_probs", "4_options/randomized/option_probs"]
    for response_type in response_types:
        for prompt_type in range(5):
            file_suffix = f"classification_response_P{prompt_type}.csv"
            filename = "/".join([directory, "processed_model_responses_cls", model_name, response_type, file_suffix])
            try:
                df = pd.read_csv(filename, dtype={'response_trimmed': 'string'})
            except Exception as e:
                # file not found: no need to process it
                print(e)
                continue

            choices_list = df["choices_list"].apply(lambda x: ast.literal_eval(x)).to_list()
            choices_list_sample = choices_list[0]

            if (type(choices_list_sample) is not list) or len(choices_list_sample) != 4:
                choices_list = df["prompt"].apply(extract_choices_list).values.tolist()

            # fix classes -- df is changed in-place and returned as well
            df = fix_classes(df, choices_list, "response_trimmed")
            na_values = df["response_trimmed"].isna()
            df.loc[na_values, "response_trimmed"] = df.loc[na_values, "original_response_trimmed"]
            # remove __fixed
            df["response_trimmed"] = df["response_trimmed"].str.strip("__fixed")
            
            df.to_csv(filename, index=False)

def filter_response(text):
    if text.lower() in ["yes", "no", "neither", "a", "b", "c", "d"]:
        return True
    else:
        return False
    

def bad_output(directory, model_name):
    response_types = ["2_options", "3_options", "4_options", "4_options/randomized", "4_options/option_probs", "4_options/randomized/option_probs"]
    for response_type in response_types:
        for prompt_type in range(5):
            file_suffix = f"classification_response_P{prompt_type}.csv"
            filename = "/".join([directory, "processed_model_responses_cls", model_name, response_type, file_suffix])
            try:
                df = pd.read_csv(filename, dtype={'response_trimmed': 'string'})
                df["response_trimmed"] = df["response_trimmed"].fillna("")
            except Exception as e:
                # file not found: no need to process it
                print(e)
                continue
            
            df["new_response"] = df["response_trimmed"].copy()
            is_proper_respose = df["response_trimmed"].apply(filter_response)
            df.loc[~is_proper_respose, "new_response"] = "Bad Output"

            df.to_csv(filename, index=False)

def save_stats(df, model_name, response_type, prompt_type, model_response_stat):
    if not model_name in model_response_stat:
        model_response_stat[model_name] = []
    
    # this is new_response, renamed in combine_all function
    if "new_response" in df.columns:
        col_name = "new_response"
    elif "response_trimmed" in df.columns:
        col_name = "response_trimmed"
    elif "text_response" in df.columns:
        col_name = "text_response"
    
    is_proper_respose = df[col_name].apply(filter_response)

    data = {
        "model_name": model_name,
        "response_type": response_type,
        "prompt_type": prompt_type,
        "num_rows": len(df),
        "proper_response": len(df[is_proper_respose]),
        "not_proper_response": len(df[~is_proper_respose]),
        "proper_response_percent": len(df[is_proper_respose])*100/len(df),
        "not_proper_response_precent": len(df[~is_proper_respose])*100/len(df),
        "new_response_counts": df[col_name].value_counts().to_dict(),
        "new_response_counts_normalized": df[col_name].value_counts(normalize=True).apply(lambda v: round(v*100, 2)).to_dict(),
    }

    model_response_stat[model_name].append(data)

def save_all_model_stats(directory, model_list, model_response_stat):
    ## Stores per model, per response type, per prompt type stats in a dict
    for model_name in model_list:
        response_types = ["2_options", "3_options", "4_options", "4_options/randomized", "4_options/option_probs", "4_options/randomized/option_probs"]
        for response_type in response_types:
            for prompt_type in range(5):
                file_suffix = f"classification_response_P{prompt_type}.csv"
                filename = "/".join([directory, model_name, response_type, file_suffix])
                try:
                    df = pd.read_csv(filename)
                except Exception as e:
                    # file not found: no need to process it
                    print(e)
                    continue
                
                save_stats(df, model_name, response_type, prompt_type, model_response_stat)

def save_all_model_stats_combined(directory, model_list, model_response_stat):
    for model_name in model_list:
        df = combine_all_responses_cls([model_name], directory)
        if len(df) > 0:
            save_stats(df, model_name, None, None, model_response_stat)

def print_total_good_bad_output_percent(combined_model_response_stat):
    total = 0
    proper = 0
    not_proper = 0
    for data in combined_model_response_stat.values():
        total += data[0]["num_rows"]
        proper += data[0]["proper_response"]
        not_proper += data[0]["not_proper_response"]

    proper_response_percent = round(proper*100/total, 2)
    not_proper_response_percent = round(not_proper*100/total, 2)
    print("proper_response_percent:", proper_response_percent)
    print("not_proper_response_percent:", not_proper_response_percent)

def parse():
    parser = argparse.ArgumentParser(
        description="Script to perform post processing on all model responses and save the combined processed file for visualization."
    )
    args_data = parser.add_argument_group(
        title="Select model directory."
    )
    args_data.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Path to directory that contains all the model repsonses to be processed.",
    )
    args_data.add_argument(
        "--do-post-process",
        action="store_true",
        default=False,
        help="Whether to perform post processing. OpenAI models don't need post processing.",
    )
    
    return parser.parse_args()


def main(directory, do_post_process):
    directory = directory.strip("/")
    if "openai" in directory:
        # openai models don't have probabilities, so they are proper model without option-probs
        include_probs = False
    else:
        include_probs = True
    model_list = get_model_list(directory)
    proper_model_list = get_cls_proper_model_list(directory, model_list, include_probs)
    fully_proper_model_list = get_fully_proper_model_list(directory, proper_model_list)

    print(len(model_list), len(proper_model_list), len(fully_proper_model_list))

    if do_post_process:
        for model_name in proper_model_list:
            # process cls
            post_process_all_responses(directory, model_name)
            # process option probs
            post_process_mcq_option_probs(directory, model_name)
            # correct mcq responses
            calibrate_mcq_responses(directory, model_name)
            # filter responses
            bad_output(directory, model_name)

    ## Save model response stats
    if do_post_process:
        save_directory = directory + "/processed_model_responses_cls"
    else:
        save_directory = directory
    
    fine_grain_model_response_stat = {}
    save_all_model_stats(save_directory, proper_model_list, fine_grain_model_response_stat)
    df = pd.DataFrame(sum(fine_grain_model_response_stat.values(), []))
    filename = f"analysis_files/{directory.replace('model_responses/', '/')}/model_output_fine_grained_summary_stats.csv"
    df.to_csv(filename, index=False)

    combined_model_response_stat = {}
    save_all_model_stats_combined(save_directory, proper_model_list, combined_model_response_stat)
    df = pd.DataFrame(sum(combined_model_response_stat.values(), []))
    filename = f"analysis_files/{directory.replace('model_responses/', '/')}/model_output_summary_stats.csv"
    df.to_csv(filename, index=False)

    print_total_good_bad_output_percent(combined_model_response_stat)
    ## CURRENT SMALL MODELS ==> 86.07 13.93
    ## CURRENT LARGE MODELS ==> 83.84 16.16 # option probs left
    ## OpenAI MODELS ==> 100.0 0.0

    ## Combine and save all processed files
    combine_all_responses(save_directory)


if __name__ == "__main__":
    args = parse()
    main(**args.__dict__)

# python3 post_process_save.py --directory model_responses/openai
# python3 post_process_save.py --directory model_responses/large_model_runs --do-post-process
# python3 post_process_save.py --directory model_responses/small_model_runs --do-post-process

# python3 combine_all_responses.py