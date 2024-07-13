# Set up

Follow https://huggingface.co/TheBloke/Platypus2-70B-Instruct-GGUF to download any of the.gguf files. 

```
platypus2-70b-instruct.Q4_K_M.gguf; Q4_K_M; 43.92 GB; medium, balanced quality - recommended
platypus2-70b-instruct.Q5_K_M.gguf; Q5_K_M; 51.25 GB; large, very low quality loss - recommended
```

Think about quality loss and memory of your GPU.

Commands used:
```bash
pip3 install huggingface-hub>=0.17.1
huggingface-cli download TheBloke/Platypus2-70B-Instruct-GGUF platypus2-70b-instruct.Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False
```

Then set up llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
pip install -r requirements.txt
```

Test on llama.cpp
```bash
./main -ngl 32 -m platypus2-70b-instruct.Q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
```

Set up llama cpp python
```bash
## Activate your VENV
set "CMAKE_ARGS=-DLLAMA_OPENBLAS=on"
set "FORCE_CMAKE=1"
pip install llama-cpp-python --no-cache-dir
```

More here: https://python.langchain.com/docs/integrations/llms/llamacpp, https://github.com/oobabooga/text-generation-webui/issues/1534

Test on llamacpp python
```python
# test_llamacpp.py
from llama_cpp import Llama

model_path = "models/platypus2-70b-instruct.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)

prompt = """system: You follow instructions. \nuser: Write a story. \nassistant:"""

output = llm(prompt, max_tokens=10, temperature=0.6, top_p=0.9)

print(output)
```

```bash
python3 test_llamacpp.py
```