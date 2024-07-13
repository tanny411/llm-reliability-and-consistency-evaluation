import time
import openai
import tiktoken
from config import openai_key, openai_org

openai.organization = openai_org
openai.api_key = openai_key

# Calculate the delay based on your rate limit: amount of time to sleep after *each* api cal
rate_limit_per_minute = 60
delay = 60.0 / rate_limit_per_minute


def call_openai_api(
    payload,
    temperature,
    max_tokens,
    logprobs,
    logit_bias_list,
    model,
    model_type,
    tries=3,
    chat_msg = None,
):
    encoding = tiktoken.encoding_for_model(model)

    def encode(text):
        encoded = encoding.encode(text)
        if len(encoded) > 1:
            raise Warning(
                f'Expected model response is more than one token. "{text}": {str(encoded)}. Only the first token will be considered.'
            )
        return str(encoded[0])

    logit_bias = {encode(val): 100 for val in logit_bias_list}

    # sleep to be within api rate limit
    time.sleep(delay + 1)

    for _try in range(tries):
        try:
            if model_type == "completion":
                response = openai.Completion.create(
                    model=model,
                    prompt=payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    logit_bias=logit_bias,
                )
            elif model_type == "chat":
                # chat-completion can handle only 1 response at a time. So batch size should be 1.
                # Also, it does not have logprobs
                if not chat_msg:
                    chat_msg = [{"role": "user", "content": payload[0]}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=chat_msg,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                )

            return response

        except Exception as e:
            print(e)
            print(f"Attempting try number {_try+1} after sleeping...")
            # https://platform.openai.com/docs/guides/rate-limits/error-mitigation
            # variation of exponential backoff instead of constant time sleep (70, 140, 210)
            time.sleep(70 * (_try + 1))
            continue

    print(f"Failed to call API after {tries} tries")
    return None
