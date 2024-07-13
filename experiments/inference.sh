#!/bin/bash

# sbatch batch-submit.sh

#SBATCH --job-name=story
#SBATCH --time=168:00:0
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
## SBATCH --nodelist=watgpu308
#SBATCH -o JOB%j_%x.out # File to which STDOUT will be written
#SBATCH -e JOB%j_%x.err # File to which STDERR will be written
#SBATCH --mail-user=aysha.kamal7@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# create conda environment, install conda_requirements.txt dependencies
# activate conda environrment before running job
source activate grs

model_list=('WizardLM/WizardMath-7B-V1.0' 'EleutherAI/gpt-j-6b' 'EleutherAI/gpt-neo-2.7B')

for model_name in ${model_list[@]};
do
   python3 -u main.py --dataset dataset/trutheval_dataset.csv --model-name $model_name --output-path model_responses/small_model_runs/ --api-type huggingface --model-type infer --temperature 0.6 --top-p 0.9 --top-k 0 --full-text-batch-size 2 --classification-batch-size 8 --run-classification --run-option-probs --run-full-text
done

# example run to use llama.cpp
# python3 -u main.py --dataset dataset/trutheval_dataset.csv --model-name garage-bAInd/Platypus2-70B-instruct --model-path models/platypus2-70b-instruct.Q4_K_M.gguf --output-path model_responses/test_run/ --api-type llamacpp --model-type infer --temperature 0.6 --top-p 0.9 --top-k 40 --run-full-text