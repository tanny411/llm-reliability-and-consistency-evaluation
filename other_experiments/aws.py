import time
import boto3
import json
from config import aws_key, aws_region, aws_secret

brt = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    aws_access_key_id = aws_key,
    aws_secret_access_key = aws_secret
)

# print(bedrock.list_foundation_models())
# meta.llama2-13b-chat-v1
# meta.llama2-70b-chat-v1
# meta.llama2-13b-v1 # not available
# meta.llama2-70b-v1 # not available


def call_aws_llama_api(
    payload,  # bs = 1
    temperature,
    max_tokens,
    logprobs,  # no
    logit_bias_list,  # no
    model,
    model_type,  # no
    tries=3,
):
    if logit_bias_list and len(logit_bias_list) > 0:
        print("Sorry, cannot use logit bias for llama models through aws!")
    if model_type != "completion":
        print(
            "Sorry, model type has to be completion. Continuing as if model_type='completion'."
        )
    if len(payload) > 1:
        print("Sorry, can only handle batch_size = 1. Taking only first value.")
    if logprobs:
        print("Sorry, can't provide logprobs.")

    payload = payload[0]

    for _try in range(tries):
        try:
            body = json.dumps(
                {
                    "prompt": payload,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    # "top_p": top_p,
                }
            )

            response = brt.invoke_model(
                body=body,
                modelId=model,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            response_text = response_body.get("generation")

            ret_response = {
                "choices": [
                    {
                        "index": 0,
                        "logprobs": None,
                        "text": response_text,
                        "text_w_input": payload + response_text,
                    }
                ]
            }

            return ret_response

        except Exception as e:
            print(e)
            print(f"Attempting try number {_try+1} after sleeping...")
            time.sleep(70 * (_try + 1))
            continue

    print(f"Failed to call API after {tries} tries")
    return None