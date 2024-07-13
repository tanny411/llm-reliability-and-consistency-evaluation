#!/bin/bash

# sbatch batch-submit.sh

#SBATCH --job-name=gpullamacpp
#SBATCH --time=72:00:0
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=watgpu308
#SBATCH -o JOB%j_%x.out # File to which STDOUT will be written
#SBATCH -e JOB%j_%x.err # File to which STDERR will be written
# SBATCH --mail-user=aysha.kamal7@gmail.com
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL

## run llama directly with llama_cpp

# cd llama.cpp
# ./main -ngl 32 -m ../models/platypus2-70b-instruct.Q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"

## run llama with llama cpp python
source activate grs
python3 test_llamacpp.py