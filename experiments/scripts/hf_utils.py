from hf_model_list import *
from utils import get_completion_type

# checks if any of the words is list l is present in the string s
contains_word = lambda s, l: any(map(lambda x: x in s, l))

sys_prompt_token = "<sys_prompt>"
prompt_token = "<prompt>"

llama2 = """\
[INST] <<SYS>>
<sys_prompt>
<</SYS>>

<prompt> [/INST]\
"""

mistral = """\
[INST] <sys_prompt> <prompt> [/INST]\
"""

mistral_openorca = """\
<|im_start|>system
<sys_prompt>
<|im_end|>
<|im_start|>user
<prompt><|im_end|>
<|im_start|>assistant
"""

identity = """\
<sys_prompt>

<prompt>
\
"""

sys_user_assistant_2 = """\
SYSTEM: <sys_prompt>
USER: <prompt>
ASSISTANT:\
"""

vicuna_one_line = "<sys_prompt> USER: <prompt> ASSISTANT: "

vicuna = """\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: Hello!
ASSISTANT: Hello!</s>
USER: How are you?
ASSISTANT: I am good.</s>
USER: <sys_prompt>
ASSISTANT: Sure!</s>
USER: <prompt>
ASSISTANT:\
"""

sys_user_assistant = """\
### System:
<sys_prompt>

### User:
<prompt>

### Assistant:
\
"""

human_assistant = """\
### Human: <sys_prompt> <prompt>
### Assistant:\
"""

sys_human_assistant = """\
<sys_prompt>
Human: <prompt>
Assistant:\
"""

prompter_assistant = "<|prompter|><sys_prompt> <prompt></s><|assistant|>"

response = """\
<sys_prompt>
<prompt>

### Response:\
"""

# full alpaca
intro_inst_inp_resp = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
<sys_prompt>

### Input:
<prompt>

### Response:
\
"""

# alpaca
inst_resp = """\
### Instruction:
<sys_prompt>
<prompt>

### Response:
\
"""

intro_inst_resp = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<sys_prompt>
<prompt>

### Response:
\
"""

inst_resp_2 = """\
Instruction: <sys_prompt> <prompt>

Response:\
"""

sys_human_bot = """\
<sys_prompt>

<human>: <prompt>
<bot>:\
"""

human_bot = """\
<human>: <sys_prompt> <prompt>
<bot>:\
"""

question_n_answer = """\
Question: <sys_prompt> <prompt>

Answer:\
"""

q_n_a = """\
Q: <sys_prompt> <prompt>
A:\
"""

only_a = """\
<sys_prompt> <prompt>
A:\
"""

pygmalion = """\
Assistant's Persona: The assistant gives helpful, detailed, and polite answers to the user's questions.
<START>

You: <sys_prompt> <prompt>
Assistant:\
"""

# for stablelm-alpha
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [50278, 50279, 50277, 1, 0]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False


stable_alpha = """\
<|SYSTEM|><sys_prompt>
<|USER|><prompt><|ASSISTANT|>\
"""

zephyr = """\
<|system|>
<sys_prompt></s>
<|user|>
<prompt></s>
<|assistant|>\
"""

prompt_types = {
    "llama2": {
        "prompt":llama2,
        "remove":["answer"],
    },
    "identity": {
        "prompt":identity,
        "remove":[],
    },
    "sys_user_assistant_2": {
        "prompt":sys_user_assistant_2,
        "remove":["all", "not_mcq_newline"],
    },
    "vicuna_one_line": {
        "prompt":vicuna_one_line,
        "remove":["answer", "not_mcq_newline"],
    },
    "vicuna": {
        "prompt":vicuna,
        "remove":["all", "not_mcq_newline"],
    },
    "sys_user_assistant": {
        "prompt":sys_user_assistant,
        "remove":["answer"],
    },
    "human_assistant": {
        "prompt":human_assistant,
        "remove":["answer", "not_mcq_newline"],
    },
    "sys_human_assistant": {
        "prompt":sys_human_assistant,
        "remove":["all", "not_mcq_newline"],
    },
    "prompter_assistant": {
        "prompt":prompter_assistant,
        "remove":["answer"],
    },
    "response": {
        "prompt":response,
        "remove":["answer"],
    },
    "intro_inst_inp_resp": {
        "prompt":intro_inst_inp_resp,
        "remove":["answer"],
    },
    "inst_resp": {
        "prompt":inst_resp,
        "remove":["answer"],
    },
    "intro_inst_resp": {
        "prompt":intro_inst_resp,
        "remove":["answer"],
    },
    "inst_resp_2": {
        "prompt":inst_resp_2,
        "remove":["all"],
    },
    "sys_human_bot": {
        "prompt":sys_human_bot,
        "remove":["answer", "not_mcq_newline"],
    },
    "human_bot": {
        "prompt":human_bot,
        "remove":["answer", "not_mcq_newline"],
    },
    "question_n_answer": {
        "prompt":question_n_answer,
        "remove":["all", "not_mcq_newline"],
    },
    "q_n_a": {
        "prompt":q_n_a,
        "remove":["all", "not_mcq_newline"],
    },
    "only_a": {
        "prompt":only_a,
        "remove":["answer", "not_mcq_newline"],
    },
    "pygmalion": {
        "prompt":pygmalion,
        "remove":["all", "not_mcq_newline"],
    },
    "stable_alpha": {
        "prompt":stable_alpha,
        "remove":["answer", "not_mcq_newline"],
    },
    "mistral": {
        "prompt": mistral,
        "remove": ["answer"]
    },
    "zephyr": {
        "prompt": zephyr,
        "remove": ["answer"]
    },
    "mistral_openorca": {
        "prompt": mistral_openorca,
        "remove": ["answer"]
    }
    
}

def format_prompt(model_name, model_type, payload, sys_prompt):
    if model_name in hf_pretrained_completion_models:
        return payload
    elif model_name in hf_finetuned_completion_model:
        prompt_type = hf_finetuned_completion_model[model_name]
        prompt_format = prompt_types[prompt_type]["prompt"]
        post_process = prompt_types[prompt_type]["remove"]
    elif model_name in hf_rl_model:
        prompt_type = hf_rl_model[model_name]
        prompt_format = prompt_types[prompt_type]["prompt"]
        post_process = prompt_types[prompt_type]["remove"]
    else:
        raise Exception(f"Model {model_name} not supported!")

    
    if "all" in post_process:
        payload = [prompt.replace("\nQuestion:", "").replace("\nChoices:", "").replace("\n\nAnswer:", "") for prompt in payload]

    if "answer" in post_process:
        payload = [prompt.replace("\n\nAnswer:", "") for prompt in payload]

    if "newline" in post_process:
        payload = [prompt.replace("\n\n", "\n").replace("\n", " ") for prompt in payload]

    if ("not_mcq_newline" in post_process) and get_completion_type(payload[0], None)!=2:
        payload = [prompt.replace("\n\n", "\n").replace("\n", " ") for prompt in payload]

    payload = [
        prompt_format.replace(sys_prompt_token, sys_prompt).replace(prompt_token, prompt)
        for prompt in payload
    ]
    
    return payload
