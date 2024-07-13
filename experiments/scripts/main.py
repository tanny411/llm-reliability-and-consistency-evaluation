import time
import shutil
import argparse
from hf_model_list import hf_model_large, hf_model_small
import pandas as pd
from api_call_utils import *
from analysis import analysis
from huggingface_api import load_hf_model
from config import TRANSFORMERS_CACHE


def parse():
    parser = argparse.ArgumentParser(
        description="Script to run dataset on a chosen LLM model with 5 specified prompts. "
        "The responses are saved in the specified directory, along with some analaysis figures."
    )
    args_data = parser.add_argument_group(
        title="Select model, folder, and other parameters."
    )
    args_data.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset.",
    )
    args_data.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model to be used.",
    )
    args_data.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .ggif model file for llama.cpp.",
    )
    args_data.add_argument(
        "--api-type",
        type=str,
        default=None,
        choices=["openai", "huggingface", "llamacpp"],
        help="Model type. One of [openai, huggingface, llamacpp]",
    )
    args_data.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["completion", "chat", "infer"],
        help="Model type. One of [completion, chat, infer]. Infer will check for the word 'chat' in model name and set it automatically.",
    )
    args_data.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to directory to save reponses and figures. For analysis, this is where the responses should be stored.",
    )
    args_data.add_argument(
        "--analysis-only",
        action="store_true",
        default=False,
        help="Perform analysis only. This assumes you already have all the responses of all completion types stored in the provided folder.",
    )
    args_data.add_argument(
        "--full-text-batch-size",
        type=int,
        default=10,
        help="Batch size for full text responses. Note: For OpenAI chat models, this will be forced to 1.",
    )
    args_data.add_argument(
        "--classification-batch-size",
        type=int,
        default=200,
        help="Batch size for classification responses. Note: For OpenAI chat models, this will be forced to 1.",
    )
    args_data.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for full text reponses. Classification responses always use temperature=0 or top_k=1 (same thing).",
    )
    args_data.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p value to pass to huggingface models. Not used with OpenAI models.",
    )
    args_data.add_argument(
        "--top-k",
        type=float,
        default=0,
        help="Top-k value to pass to huggingface models. Not used with OpenAI models. Classification responses always use temperature=0 or top_k=1 (same thing).",
    )
    args_data.add_argument(
        "--run-classification",
        action="store_true",
        default=False,
        help="Runs 2 options, 3 options, 4 options, and 4 options randomized classifications.",
    )
    args_data.add_argument(
        "--run-full-text",
        action="store_true",
        default=False,
        help="Gets full text responses.",
    )
    args_data.add_argument(
        "--run-option-probs",
        action="store_true",
        default=False,
        help="Runs 4 options and 4 options randomized classifications and gets only the probabilities of A,B,C,D.",
    )

    return parser.parse_args()


def main(
    dataset,
    model_name,
    model_path,
    api_type,
    model_type,
    output_path,
    analysis_only,
    full_text_batch_size,
    classification_batch_size,
    temperature,
    top_p,
    top_k,
    run_classification,
    run_full_text,
    run_option_probs,
):
    if model_type == "infer":
        if "chat" in model_name:
            model_type = "chat"
        else:
            model_type = "completion"

    if not output_path.endswith("/"):
        output_path += "/"

    folder_name = model_name.replace("/", "--")
    output_path += folder_name + "/"

    # if analysis_only:
    #     analysis(output_path)
    #     return

    check_and_create_folder(output_path)

    if is_openai_chat(model_type, api_type) or api_type=="llamacpp":
        full_text_batch_size = 1
        classification_batch_size = 1

    print("NOW RUNNING:", model_name)
    print("CLASSIFICATION BATCH SIZE:", classification_batch_size)
    print("FULL TEXT BATCH SIZE:", full_text_batch_size)

    if api_type == "huggingface":
        model = load_hf_model(model_name)
    elif api_type == "llamacpp":
        from llama_cpp import Llama
        # model_path should be path of the .gguf model file
        # model_name should be the model's hf name
        if not model_path:
            raise Exception("model_path not found. `.gguf` file path required for llama.cpp")
        model = Llama(model_path=model_path, n_gpu_layers=-1, logits_all=False, verbose=False)
    elif api_type == "openai":
        model = None

    # Load Data
    # "../../../curated_datasets/version_3/data_v3.csv"
    data = pd.read_csv(dataset)

    if run_classification:
        get_classification_response(
            data=data,
            main_directory=output_path,
            model_name=model_name,
            model=model,
            model_type=model_type,
            api_type=api_type,
            batch_size=classification_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        get_randomized_mcq_classification_response(
            data=data,
            main_directory=output_path,
            model_name=model_name,
            model=model,
            model_type=model_type,
            api_type=api_type,
            batch_size=classification_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    if run_full_text:
        get_full_text_response(
            data=data,
            main_directory=output_path,
            model_name=model_name,
            model=model,
            model_type=model_type,
            api_type=api_type,
            batch_size=full_text_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # can get option probs only for hf models
    if api_type == "huggingface" and run_option_probs:
        get_mcq_response_with_options_prob(
            data=data,
            main_directory=output_path,
            model_name=model_name,
            model=model,
            model_type=model_type,
            api_type=api_type,
            batch_size=classification_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    analysis(output_path)

    if api_type == "huggingface":
        shutil.rmtree(TRANSFORMERS_CACHE + "/models--" + folder_name)


def run_all_models():
    run_models = list(hf_model_small.keys())
    # print(run_models)

    for model_name in run_models:
        try:
            main(
                "../../../curated_datasets/version_3/data_v3.csv",
                model_name,
                "huggingface",
                "infer",
                "model_responses/new_run_all_models/",
                False,
                2,
                8,
                0.6,
                0.9,
                0,
                True,
                True,
                True,
            )
        except Exception as e:
            print("########## FAILED ##########")
            print(model_name)
            print(e)


def run_all_models_open_ai():
    for model_name, model_type in run_models_open_ai:
        try:
            main(
                "../../../curated_datasets/version_3/data_v3.csv",
                model_name,
                "openai",
                model_type,
                "model_responses/",
                False,
                10,
                200,
                0.7,
                0.9,
                0,
                True,
                True,
                True,
            )
        except Exception as e:
            print("########## FAILED ##########")
            print(model_name)
            print(e)


if __name__ == "__main__":
    # run_all_models_open_ai()
    # run_all_models()
    args = parse()
    main(**args.__dict__)
