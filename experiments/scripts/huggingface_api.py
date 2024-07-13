import os
import warnings
from config import *
from hf_utils import *

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Cuda set based on what's available after checking nvidia-smi
# the chosen devices are then numbered as 0,1,2,... by pytorch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

def set_device():
    if cuda.is_available():
        # set it manually if you want to access a free GPU
        # cuda_current_device = cuda.current_device()
        # device = f"cuda:{cuda_current_device}"
        device = "cuda:0"
        device_name = torch.cuda.get_device_name()
        print(f"Using device: {device} ({device_name})")
    else:
        device = "cpu"
        print(f"Using device: {device}")

    return device


def load_hf_model(model_name):
    device = set_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # add_prefix_space=True, # for sequence bias
        use_auth_token=hg_token,
        padding_side="left",
        legacy=False,
    )

    # Use of device_map: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    # device_map="auto" starts using all GPUs available, "sequential" will use later GPUs only if required
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hg_token,
        trust_remote_code=True,
        # device_map="balanced_low_0",
        # offload_folder=".offload",
        # load_in_4bit=True
    ).to(device)  # remove .to(device) with device_map

    model.eval()

    if not model.config.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, device


def call_hf_api(
    payload,
    sys_prompt,
    temperature,
    top_p,
    top_k,
    max_tokens,
    logit_bias_list,
    model_name,
    model_type,
    loaded_model,
    get_option_probs,
):
    model, tokenizer, device = loaded_model

    inputs = tokenizer(
        format_prompt(model_name, model_type, payload, sys_prompt),
        padding=True,
        return_tensors="pt",
    ).to(device)

    # def get_tokens_as_tuple(word):
    #     return tuple(tokenizer([word], add_special_tokens=False).input_ids[0])

    # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    # Forbidding a sequence is equivalent to setting its bias to -float("inf")
    # if len(logit_bias_list) == 0:
    #     sequence_bias = None
    # else:
    #     sequence_bias = {get_tokens_as_tuple(val): 100.0 for val in logit_bias_list}

    # We will always return logprobs
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        # False for greedy setting, top_k=1 (getting the highest probability token)
        # Sampling sets too many probs to 100% for some reason
        do_sample=top_k != 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,  # activate sampling and deactivate top_k by setting top_k sampling to 0
        # sequence_bias=sequence_bias,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        # begin_suppress_tokens = [0], # "<unk>" = 0 # not for all models
        # suppress_tokens = [0],
        renormalize_logits=True,
        # remove_invalid_values=True,  # because we get errors with nan and infs, but this gives nonsense outputs # happens when I want to use more than 1 GPU
    )

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    generated_text = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = tokenizer.batch_decode(outputs.sequences)

    # print(generated_tokens, transition_scores, text)
    # print(text[0])
    # for tok, prob in zip(generated_tokens[0], transition_scores[0]):
    #     print("Token:", tok, "Token text:",  tokenizer.decode(tok), "Prob:", prob)
    
    # Normalize score across vocab dimension
    scores = outputs.scores # [# generated steps, batch_size, vocab_size]
    # reshape scores as [batch_size*vocab_size, # generation steps]
    # with # generation steps being seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
    scores = scores.reshape(-1, model.config.vocab_size, scores.shape[-1]) # [batch_size, vocab_size, # generated steps]
    scores = torch.nn.functional.log_softmax(scores, dim=1)

    response = {"choices": []}
    for i in range(len(payload)):
        response_i = {
            "index": i,
            "logprobs": {"tokens": [], "token_logprobs": []},
            "text": generated_text[i],
            "text_w_input": text[i],
        }
        if get_option_probs:
            for token in ["A", "B", "C", "D"]:
                token_id = tokenizer([token], add_special_tokens=False).input_ids[0][0]
                score = scores[i,token_id,0] # max_token = 1, so we only care about the first token
                response_i["logprobs"]["tokens"].append(token)
                response_i["logprobs"]["token_logprobs"].append(score.item())
        else:
            for tok, score in zip(generated_tokens[i], transition_scores[i]):
                if tok not in tokenizer.all_special_ids:
                    token = tokenizer.decode(tok)
                    if token != "":
                        response_i["logprobs"]["tokens"].append(token)
                        response_i["logprobs"]["token_logprobs"].append(score.item())

        response["choices"].append(response_i)

    return response
