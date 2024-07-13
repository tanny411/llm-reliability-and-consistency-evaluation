from hf_utils import *
import time

# only supports full text for now
# needs modification to run cls and option_probs

def call_llamacpp_api(
    payload,
    sys_prompt,
    temperature,
    top_p,
    top_k, # will use default values
    max_tokens,
    logprobs, # set True when model was initialized
    logit_bias, # not impleneted # will not be either # not used
    model_name,
    model,
    model_type,
    get_options_prob, # not implemented
):
    ## only handle full_text for now
    start = time.time()
    prompt = format_prompt(model_name, model_type, payload, sys_prompt)[0]
    time_taken = (time.time() - start) / 60
    print("It took {0:0.1f} minutes to format prompt.".format(time_taken))

    start = time.time()
    model.reset()
    output = model(prompt, max_tokens=100, temperature=temperature, top_p=top_p)
    print(output)

    time_taken = (time.time() - start) / 60
    print("It took {0:0.1f} minutes to generate output.".format(time_taken))

    return output
