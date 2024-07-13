# values are prompt format string that need <prompt> and <sys_prompt> replacement
# values may additionally contain a post-processing function that needs to be applied after the former replacement4
"""
Steps to choose a models:
- High performing models in HF leaderboard
- Their smaller and other variants
- Newer versions
- Models with most data/ training datasets. Ignored the ones that were subsumed by other models
- Instruct and Chat present, then chose instruct
- Some were removed because of just bad performance (gpt-j-6b)
- Likes in HF
"""
hf_model_small = {
    # FT
    ## DONE ## "migtissera/Synthia-7B": "sys_user_assistant_2",  # 57.53
    ## DONE ## "databricks/dolly-v2-7b": "identity",  # 43.56
    ## DONE ## "databricks/dolly-v2-3b": "identity",
    ## DONE ## "togethercomputer/RedPajama-INCITE-7B-Instruct": "q_n_a",  # 46.9
    ## DONE ## "mosaicml/mpt-7b-instruct": "intro_inst_resp",  # 48.92
    ## DONE ##  "tiiuae/falcon-7b-instruct": "sys_human_assistant",  # 46.58
    ## DONE ##  "facebook/opt-iml-max-1.3b": "identity",  # fine-tuned
    # "stabilityai/StableBeluga-7B": "sys_user_assistant",  # 59.59 # llama2 + orca
    # "quantumaikr/llama-2-7b-hf-guanaco-1k": "sys_user_assistant",  # 55.15
    # "Mikael110/llama-2-7b-guanaco-fp16": "sys_human_assistant",  # 56.18
    ## CANCEL ## "WizardLM/WizardMath-7B-V1.0": "intro_inst_resp",  # 55.82
    # "h2oai/h2ogpt-gm-oasst1-en-1024-open-llama-7b-preview-400bt": "identity",  # 43.32 ***supposed to auto format prompt***
    ## DONE ## "h2oai/h2ogpt-oig-oasst1-512-6_9b": "sys_human_bot",  # 41.9 # pythia-6.9b +  h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v1 and h2oai/openassistant_oasst1_h2ogpt and h2oai/h2ogpt-fortune2000-personalized and h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v3
    ## DONE ## "lmsys/vicuna-7b-v1.5": "vicuna",  # llama2 - vicuna
    # "PygmalionAI/pygmalion-350m": "pygmalion",  # 32.22
    # "PygmalionAI/pygmalion-1.3b": "pygmalion",  # 34.2
    # "PygmalionAI/pygmalion-2.7b": "pygmalion",  # 36.84
    ## DONE ## "PygmalionAI/pygmalion-6b": "pygmalion",  # 43.03
    ## DONE ## "Neko-Institute-of-Science/pygmalion-7b": "pygmalion",  # 49.85
    ## DONE ## "mistralai/Mistral-7B-Instruct-v0.1": "mistral", # 60.97
    ## DONE ## "HuggingFaceH4/zephyr-7b-alpha": "zephyr", # fine tuned on mistral base # 60.17
    ## DONE ## "Open-Orca/Mistral-7B-OpenOrca": "mistral_openorca", # fine tuned on mistral base # 59.5
    # RL
    ## DONE ## "meta-llama/Llama-2-7b-chat-hf": "llama2",
    # BASE
    ## DONE ## "meta-llama/Llama-2-7b-hf": None,
    # "huggyllama/llama-7b": None,
    ## DONE ## "tiiuae/falcon-7b": None,
    ## CANCEL ## "mosaicml/mpt-7b": None,
    # "openlm-research/open_llama_3b": None,  # v1
    # "openlm-research/open_llama_7b": None,  # v1
    ## DONE ## "openlm-research/open_llama_3b_v2": None,  # v2
    ## DONE ## "openlm-research/open_llama_7b_v2": None,  # v2
    # "facebook/opt-6.7b": None,
    # "facebook/opt-2.7b": None,
    # "facebook/opt-1.3b": None,
    # "facebook/opt-350m": None,
    # "facebook/opt-125m": None,
    ## CANCEL ## "EleutherAI/gpt-neo-2.7B": None,
    # "EleutherAI/gpt-neo-1.3B": None,
    # "EleutherAI/gpt-neo-125m": None,
    ## CANCEL ## "EleutherAI/gpt-j-6b": None,
    # "EleutherAI/pythia-70m-deduped": None,
    # "EleutherAI/pythia-160m-deduped": None,
    # "EleutherAI/pythia-410m-deduped": None,
    # "EleutherAI/pythia-1b-deduped": None,
    # "EleutherAI/pythia-1.4b-deduped": None,
    # "EleutherAI/pythia-2.8b-deduped": None,
    ## DONE ## "EleutherAI/pythia-6.9b-deduped": None,
    # "cerebras/Cerebras-GPT-111M": None,
    # "cerebras/Cerebras-GPT-256M": None,
    # "cerebras/Cerebras-GPT-590M": None,
    # "cerebras/Cerebras-GPT-1.3B": None,
    # "cerebras/Cerebras-GPT-2.7B": None,
    ## CANCEL ## "cerebras/Cerebras-GPT-6.7B": None,
    ## CANCEL ## "gpt2-xl": None,  # 1.5B params
    # "gpt2-large": None,  # 774M params
    # "gpt2-medium": None,  # 355M params
    # "gpt2": None,  # 124M params
    # "bigscience/bloom-560m": None,
    # "bigscience/bloomz-560m": None,  # zero shot QnA (make answer section clear)
    # "bigscience/bloom-1b1": None,
    # "bigscience/bloomz-1b1": None,
    # "bigscience/bloom-3b": None,
    # "bigscience/bloomz-3b": None,
    # "bigscience/bloom-7b1": None,
    ## DONE ## "bigscience/bloomz-7b1-mt": None,
    ## DONE ## "togethercomputer/RedPajama-INCITE-7B-Base": None,
    ## DONE ## "mistralai/Mistral-7B-v0.1": None,
}
hf_model_large = {
    # FT
    "migtissera/Synthia-70B": "sys_user_assistant_2",  # 71.32
    ## DONE ## "migtissera/Synthia-13B": "sys_user_assistant_2",  # 61.34
    ## DONE ## "databricks/dolly-v2-12b": "identity",  # 43.67
    "togethercomputer/GPT-NeoXT-Chat-Base-20B": "sys_human_bot",  # 46.03 # can use few-shot
    "tiiuae/falcon-40b-instruct": "sys_human_assistant",  # 63.47
    # "TheBloke/stable-vicuna-13B-HF": "human_assistant",  # 57.63 # RLHF on vicuna, which is finetuned on llama
    ## DONE ## "facebook/opt-iml-max-30b": "identity",  # fine-tuned
    "garage-bAInd/Platypus2-70B-instruct": "inst_resp",  # 69.3 # merge of garage-bAInd/Platypus2-70B and upstage/Llama-2-70b-instruct-v2
    # "upstage/SOLAR-0-70b-16bit": "sys_user_assistant",  # 72.95 # was upstage/Llama-2-70b-instruct-v2 # llama2 + orca + alpaca
    ## DONE ## "garage-bAInd/Stable-Platypus2-13B": "inst_resp",  # 63.96 # Platypus2-13B + Stable Beluga
    # "garage-bAInd/Platypus2-70B": "inst_resp",  # 70.06 # llama2 + STEM and logic based dataset garage-bAInd/Open-Platypus
    # "garage-bAInd/Platypus-30B": "inst_resp",  # 64.61
    # "garage-bAInd/Platypus2-13B": "inst_resp",  # 61.35
    "stabilityai/StableBeluga2": "sys_user_assistant",  # 67.42  # llama2 + orca # 70B models
    # "stabilityai/StableBeluga-13B": "sys_user_assistant",  # 62.91 # llama2 + orca
    # "dfurman/llama-2-70b-dolphin-peft": "identity",  # 70.76 # llama2 + dolphin-25k
    # "quantumaikr/llama-2-70b-fb16-guanaco-1k": "sys_user_assistant",  # 71.41 # llama2 + guanaco-llama2-1k
    # "TheBloke/llama-2-70b-Guanaco-QLoRA-fp16": "sys_human_assistant",  # 70.63 # llama2 + guanaco
    # "Mikael110/llama-2-13b-guanaco-fp16": "sys_human_assistant",  # 60.67
    # "jordiclive/Llama-2-70b-oasst-1-200": "prompter_assistant",  # 69.04 # llama2 + OASST
    # "ddobokki/Llama-2-70b-orca-200k": "human_assistant",  # 68.29 # llama2 + open-orca
    "WizardLM/WizardMath-70B-V1.0": "intro_inst_resp", # 60.42
    "WizardLM/WizardMath-13B-V1.0": "intro_inst_resp",
    # "WizardLM/WizardLM-13B-V1.2": "vicuna",  # 60.79 # llama2 (full weights)
    # "WizardLM/WizardLM-70B-V1.0": "vicuna",  # llama2
    # "TheBloke/gpt4-alpaca-lora-13B-HF": "inst_resp_2",  # 59.52
    # "TheBloke/gpt4-alpaca-lora-30b-HF": "inst_resp_2",  ## This is a pre-merged version of the Chansung GPT4 Alpaca 30B LoRA model.
    # "TheBloke/gpt4-alpaca-lora_mlp-65B-HF": "inst_resp_2",  # 68.26 # llama + alpaca_data_gpt4
    "CalderaAI/30B-Lazarus": "intro_inst_inp_resp",  # 66.08 # llama + medley of various models
    # "h2oai/h2ogpt-research-oasst1-llama-65b": "sys_human_bot",  # llama + h2oai/openassistant_oasst1_h2ogpt_graded dataset
    # "h2oai/h2ogpt-research-oig-oasst1-512-30b": "sys_human_bot",  # llama 30 + h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v2
    # "h2oai/h2ogpt-gm-oasst1-en-1024-20b": "identity",  # 46.67 ***supposed to auto format prompt*** # gpt-neox-20b + oasst1
    # "h2oai/h2ogpt-oasst1-512-12b": "sys_human_bot",  # 43.75 # pythia-12b + h2oai/openassistant_oasst1_h2ogpt_graded
    "lmsys/vicuna-13b-v1.5": "vicuna",  # llama2 - vicuna
    "Open-Orca/OpenOrca-Platypus2-13B": "inst_resp",  # 64.56 # llama2 + platypus + open-orca
    "TehVenom/Pygmalion-13b-Merged": "pygmalion",  # 53.83
    # RL
    "meta-llama/Llama-2-70b-chat-hf": "llama2",
    ## DONE ## "meta-llama/Llama-2-13b-chat-hf": "llama2",
    # "tiiuae/falcon-180B-chat": "sys_human_assistant", # 67.85
    # BASE
    "meta-llama/Llama-2-70b-hf": None, # 67.87
    "meta-llama/Llama-2-13b-hf": None,
    # "huggyllama/llama-65b": None,
    # "huggyllama/llama-13b": None,
    # "tiiuae/falcon-180B": None,  # 180B too large to be easily handled with transformers+acccelerate, we recommend using Text Generation Inference.
    "tiiuae/falcon-40b": None,
    "mosaicml/mpt-30b": None,
    # "openlm-research/open_llama_13b": None,  # v1
    # "facebook/opt-66b": None,
    # "facebook/opt-30b": None,
    # "facebook/opt-13b": None,
    "EleutherAI/gpt-neox-20b": None,
    "EleutherAI/pythia-12b-deduped": None,
    "cerebras/Cerebras-GPT-13B": None,
}

# this list is divided into small and large above. this is kept to keep note
# of the types of models used overall
hf_pretrained_completion_models = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-hf",
    "huggyllama/llama-65b",
    "huggyllama/llama-13b",
    "huggyllama/llama-7b",
    "tiiuae/falcon-180B",  # 180B too large to be easily handled with transformers+acccelerate, we recommend using Text Generation Inference.
    "tiiuae/falcon-40b",
    "tiiuae/falcon-7b",
    "mosaicml/mpt-30b",
    "mosaicml/mpt-7b",
    "openlm-research/open_llama_3b",  # v1
    "openlm-research/open_llama_7b",  # v1
    "openlm-research/open_llama_13b",  # v1
    "openlm-research/open_llama_3b_v2",  # v2
    "openlm-research/open_llama_7b_v2",  # v2
    "facebook/opt-66b",
    "facebook/opt-30b",
    "facebook/opt-13b",
    "facebook/opt-6.7b",
    "facebook/opt-2.7b",
    "facebook/opt-1.3b",
    "facebook/opt-350m",
    "facebook/opt-125m",
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "cerebras/Cerebras-GPT-111M",
    "cerebras/Cerebras-GPT-256M",
    "cerebras/Cerebras-GPT-590M",
    "cerebras/Cerebras-GPT-1.3B",
    "cerebras/Cerebras-GPT-2.7B",
    "cerebras/Cerebras-GPT-6.7B",
    "cerebras/Cerebras-GPT-13B",
    # "stabilityai/stablelm-base-alpha-7b-v2",
    # "stabilityai/stablelm-base-alpha-3b-v2",
    # "stabilityai/stablelm-base-alpha-7b",
    # "stabilityai/stablelm-base-alpha-3b",
    "gpt2-xl",  # 1.5B params
    "gpt2-large",  # 774M params
    "gpt2-medium",  # 355M params
    "gpt2",  # 124M params
    "bigscience/bloom-560m",
    "bigscience/bloomz-560m",  # zero shot QnA (make answer section clear)
    "bigscience/bloom-1b1",
    "bigscience/bloomz-1b1",
    "bigscience/bloom-3b",
    "bigscience/bloomz-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloomz-7b1-mt",
    "togethercomputer/RedPajama-INCITE-7B-Base",
    # "BlinkDL/rwkv-4-world",  # RNN model
    ## NEW
    "mistralai/Mistral-7B-v0.1"
]

# this list is divided into small and large above. this is kept to keep note
# of the types of models used overall
hf_finetuned_completion_model = {
    "migtissera/Synthia-70B": "sys_user_assistant_2",  # 71.32
    "migtissera/Synthia-13B": "sys_user_assistant_2",  # 61.34
    "migtissera/Synthia-7B": "sys_user_assistant_2",  # 57.53
    "databricks/dolly-v2-12b": "identity",  # 43.67
    "databricks/dolly-v2-7b": "identity",  # 43.56
    "databricks/dolly-v2-3b": "identity",
    "togethercomputer/GPT-NeoXT-Chat-Base-20B": "sys_human_bot",  # 46.03 # can use few-shot
    # "togethercomputer/GPT-JT-6B-v1": "only_a",  # 48.07 # improves the performance of classification over GPT-J # does not produce proper formatted response
    "togethercomputer/RedPajama-INCITE-7B-Instruct": "q_n_a",  # 46.9
    # "togethercomputer/RedPajama-INCITE-7B-Chat": "human_bot",  # 43.98
    "mosaicml/mpt-7b-instruct": "intro_inst_resp",  # 48.92
    # "mosaicml/mpt-7b-chat": "identity",  # 49.95
    # "stabilityai/stablelm-tuned-alpha-7b": "stable_alpha",  # 37.57 # old v
    # "stabilityai/stablelm-tuned-alpha-3b": "stable_alpha",  # old v # 34.32
    "TheBloke/stable-vicuna-13B-HF": "human_assistant",  # 57.63 # RLHF on vicuna, which is finetuned on llama
    "tiiuae/falcon-40b-instruct": "sys_human_assistant",  # 63.47
    "tiiuae/falcon-7b-instruct": "sys_human_assistant",  # 46.58
    # "facebook/opt-iml-1.3b": "identity",  # fine-tuned
    # "facebook/opt-iml-30b": "identity",  # fine-tuned
    "facebook/opt-iml-max-1.3b": "identity",  # fine-tuned
    "facebook/opt-iml-max-30b": "identity",  # fine-tuned
    "upstage/SOLAR-0-70b-16bit": "sys_user_assistant",  # 72.95 # was upstage/Llama-2-70b-instruct-v2 # llama2 + orca + alpaca
    "garage-bAInd/Platypus2-70B-instruct": "inst_resp",  # 73.13 # merge of garage-bAInd/Platypus2-70B and upstage/Llama-2-70b-instruct-v2
    "garage-bAInd/Stable-Platypus2-13B": "inst_resp",  # 63.96 # Platypus2-13B + Stable Beluga
    "garage-bAInd/Platypus2-70B": "inst_resp",  # 70.06 # llama2 + STEM and logic based dataset garage-bAInd/Open-Platypus
    "garage-bAInd/Platypus-30B": "inst_resp",  # 64.61
    "garage-bAInd/Platypus2-13B": "inst_resp",  # 61.35
    # "upstage/llama-65b-instruct": "sys_user_assistant",  # 69.94 # llama + orca
    # "upstage/llama-30b-instruct-2048": "sys_user_assistant",  # 67.02 # llama + openbookqa + sciq + Open-Orca/OpenOrca + metaeval/ScienceQA_text_only + GAIR/lima
    "stabilityai/StableBeluga2": "sys_user_assistant",  # 71.42  # llama2 + orca # 70B models
    "stabilityai/StableBeluga-13B": "sys_user_assistant",  # 62.91 # llama2 + orca
    "stabilityai/StableBeluga-7B": "sys_user_assistant",  # 59.59 # llama2 + orca
    # "stabilityai/StableBeluga1-Delta",  # llama (65B) + orca (delta weights)
    # "jondurbin/airoboros-l2-70b-gpt4-1.4.1": "vicuna_one_line",  # 70.93 # llama2 + airoboros-gpt4-1.4.1 dataset (gpt4 generated dataset)
    # "jondurbin/airoboros-65b-gpt4-1.2": "vicuna_one_line",  # 67.01 # there is a 1.4 # llama + airboros
    # "jondurbin/airoboros-33b-gpt4-1.4": "vicuna_one_line",  # 64.89 # there is a 1.4 # llama + airboros
    # "jondurbin/airoboros-13b": "vicuna_one_line",  # 60.23
    "dfurman/llama-2-70b-dolphin-peft": "identity",  # 70.76 # llama2 + dolphin-25k
    "quantumaikr/llama-2-70b-fb16-guanaco-1k": "sys_user_assistant",  # 71.41 # llama2 + guanaco-llama2-1k
    "quantumaikr/llama-2-7b-hf-guanaco-1k": "sys_user_assistant",  # 55.15
    "TheBloke/llama-2-70b-Guanaco-QLoRA-fp16": "sys_human_assistant",  # 70.63 # llama2 + guanaco
    "Mikael110/llama-2-13b-guanaco-fp16": "sys_human_assistant",  # 60.67
    "Mikael110/llama-2-7b-guanaco-fp16": "sys_human_assistant",  # 56.18
    # "TheBloke/guanaco-65B-HF": "sys_human_assistant",  # 66.91 # fp16
    # "timdettmers/guanaco-65b-merged": "sys_human_assistant",  # llama + guanaco # ^^ same
    # "timdettmers/guanaco-33b-merged": "sys_human_assistant",
    # "TheBloke/guanaco-13B-HF": "sys_human_assistant",  # 59.18
    # "TheBloke/guanaco-7B-HF": "sys_human_assistant",  # 51.89
    "jordiclive/Llama-2-70b-oasst-1-200": "prompter_assistant",  # 69.04 # llama2 + OASST
    "ddobokki/Llama-2-70b-orca-200k": "human_assistant",  # 68.29 # llama2 + open-orca
    "WizardLM/WizardMath-70B-V1.0": "intro_inst_resp",
    "WizardLM/WizardMath-13B-V1.0": "intro_inst_resp",
    "WizardLM/WizardMath-7B-V1.0": "intro_inst_resp",  # 55.82
    # "WizardLM/WizardLM-7B-V1.0", # llama (delta weights)
    # "WizardLM/WizardLM-13B-V1.0", # llama (delta weights)
    # "WizardLM/WizardLM-30B-V1.0", # llama (delta weights)
    # "WizardLM/WizardLM-13B-V1.1": "vicuna",  # 61.78 # not sure about prompt format # llama (full weights)
    "WizardLM/WizardLM-13B-V1.2": "vicuna",  # 60.79 # llama2 (full weights)
    "WizardLM/WizardLM-70B-V1.0": "vicuna",  # llama2
    # "chansung/gpt4-alpaca-lora-30b",  # llama + alpaca # LoRA weights
    # "chansung/gpt4-alpaca-lora-13b",  # llama + alpaca # LoRA weights
    # "chansung/gpt4-alpaca-lora-7b",  # llama + alpaca # LoRA weights
    "TheBloke/gpt4-alpaca-lora-13B-HF": "inst_resp_2",  # 59.52
    "TheBloke/gpt4-alpaca-lora-30b-HF": "inst_resp_2",  ## This is a pre-merged version of the Chansung GPT4 Alpaca 30B LoRA model.
    "TheBloke/gpt4-alpaca-lora_mlp-65B-HF": "inst_resp_2",  # 68.26 # llama + alpaca_data_gpt4
    "CalderaAI/30B-Lazarus": "intro_inst_inp_resp",  # 66.08 # llama + medley of various models
    "h2oai/h2ogpt-research-oasst1-llama-65b": "sys_human_bot",  # llama + h2oai/openassistant_oasst1_h2ogpt_graded dataset
    "h2oai/h2ogpt-research-oig-oasst1-512-30b": "sys_human_bot",  # llama 30 + h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v2
    "h2oai/h2ogpt-gm-oasst1-en-1024-20b": "identity",  # 46.67 ***supposed to auto format prompt*** # gpt-neox-20b + oasst1
    "h2oai/h2ogpt-oasst1-512-12b": "sys_human_bot",  # 43.75 # pythia-12b + h2oai/openassistant_oasst1_h2ogpt_graded
    "h2oai/h2ogpt-gm-oasst1-en-1024-open-llama-7b-preview-400bt": "identity",  # 43.32 ***supposed to auto format prompt***
    "h2oai/h2ogpt-oig-oasst1-512-6_9b": "sys_human_bot",  # 41.9 # pythia-6.9b +  h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v1 and h2oai/openassistant_oasst1_h2ogpt and h2oai/h2ogpt-fortune2000-personalized and h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v3
    "lmsys/vicuna-7b-v1.5": "vicuna",  # llama2 - vicuna
    "lmsys/vicuna-13b-v1.5": "vicuna",  # llama2 - vicuna
    # "lmsys/vicuna-7b-v1.3": "vicuna",  # 55.62 # llama - vicuna
    # "lmsys/vicuna-13b-v1.3": "vicuna",  # 60.01 # llama - vicuna
    # "lmsys/vicuna-33b-v1.3": "vicuna",  # llama - vicuna
    "Open-Orca/OpenOrca-Platypus2-13B": "inst_resp",  # 64.56 # llama2 + platypus + open-orca
    "PygmalionAI/pygmalion-350m": "pygmalion",  # 32.22
    "PygmalionAI/pygmalion-1.3b": "pygmalion",  # 34.2
    "PygmalionAI/pygmalion-2.7b": "pygmalion",  # 36.84
    "PygmalionAI/pygmalion-6b": "pygmalion",  # 43.03
    "Neko-Institute-of-Science/pygmalion-7b": "pygmalion",  # 49.85
    "TehVenom/Pygmalion-13b-Merged": "pygmalion",  # 53.83
    ## NEW
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral", # 60.97
    "HuggingFaceH4/zephyr-7b-alpha": "zephyr", # fine tuned on mistral base # 60.17
    "Open-Orca/Mistral-7B-OpenOrca": "mistral_openorca", # fine tuned on mistral base # 59.5
}

# this list is divided into small and large above. this is kept to keep note
# of the types of models used overall
hf_rl_model = {
    "meta-llama/Llama-2-70b-chat-hf": "llama2",
    "meta-llama/Llama-2-13b-chat-hf": "llama2",
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "tiiuae/falcon-180B-chat": "sys_human_assistant",
}
