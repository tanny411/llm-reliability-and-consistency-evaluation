# Uncovering the Reliability and Consistency of AI Language Models: A Systematic Study

Author: Aisha Khatun\
Affiliation: David R. Cheriton School of Computer Science, University of Waterloo, Canada.\
Thesis: [Thesis PDF Link](https://uwspace.uwaterloo.ca/items/e01e11a6-e033-4f6a-85c6-849fba74e039) \
Slides: [Google Slides Link](https://docs.google.com/presentation/d/1rzBgxt7JzPmmHgOkzHOwa6HNvHF5m50eMRWxGJavtoY/edit?usp=sharing) \
This thesis is presented to the University of Waterloo in fulfillment of the thesis requirement for the degree of Master of Mathematics in Computer Science.

Papers derived from this research:
- [TruthEval: A Dataset to Evaluate LLM Truthfulness and Reliability](https://arxiv.org/abs/2406.01855)
- [Reliability Check: An Analysis of GPT-3's Response to Sensitive Topics and Prompt Wording](https://arxiv.org/abs/2306.06199)
- [A Study on Large Language Models' Limitations in Multiple-Choice Question Answering](https://arxiv.org/abs/2401.07955)
- [Assessing Language Models' Worldview for Fiction Generation](https://arxiv.org/abs/2408.07904)

## Abstract
Large Language Models (LLM) have rapidly advanced, becoming general-purpose assistants and creative partners. Despite their widespread use, LLMs exhibit significant vulnerabilities to prompt variations and struggle with task understanding, leading to inconsistencies and factual inaccuracies in their responses. Traditional Natural Language Processing (NLP) benchmarks often overlook nuances in LLM behavior and reliability. This thesis addresses this gap by curating a dataset across six categories: Fact, Conspiracy, Controversy, Misconception, Stereotype, and Fiction. We rigorously define LLMs' factual accuracy, consistency, and robustness to prompt variations using diverse response formats and question variations, and evaluate these on 37 models. Our findings reveal LLMs' volatility and unreliability, particularly in the Controversy and Misconception categories, where conflicting training data impedes performance. Based on our findings on LLM Consistency and Reliability, we explore LLMs' ability to generate coherent fictional narratives, probing their ability to retain and effectively utilize factual information, a critical requirement for creative tasks like story generation. While LLMs offer versatile applications, their reliability hinges on addressing challenges in prompt understanding and response consistency, emphasizing the need for ongoing research to enhance their performance across diverse tasks and applications.

# What's in the repository?

## Datset
The dataset is oficcially published here: [Borealis](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2F5MZWBV)

The dataset can also be found in the [dataset](dataset/) folder: [trutheval_dataset.csv](dataset/trutheval_dataset.csv). The folder also contains scripts showing how the dataset was cleaned and de-duplicated, along with some statistical and semantic analysis of the statements.


## Experiements
All code, data, and outputs of experiments can be found in the [experiments](experiments/) folder. Additional experiments (e.g. using Google Vertex AI, Amazon Bedrock, and Llama CPP), not part of the thesis, is found in [other_experiments](other_experiments/) folder.

Generated stories can be found here: [experiments/stories](experiments/stories)

Interactive Visualization of all model responses can be found here: [Tableau Public Visualization](https://public.tableau.com/app/profile/aisha.khatun/viz/LLMReliabilityandConsistencyAnalysis/Sheet1a)

### How to run the code?
- Create a conda environment and install `conda_requirements.txt` dependencies.
- Run `inference.sh` to generate responses from a list of models as defined in the script.
- Run the following code to post-process the responses.
    ```bash
        python3 post_process_save.py --directory model_responses/openai
        python3 post_process_save.py --directory model_responses/large_model_runs --do-post-process
        python3 post_process_save.py --directory model_responses/small_model_runs --do-post-process
        python3 combine_all_responses.py
    ```
- To generate stories, use `generate_stories.py` file. The model name needs to be changed within the file.

### Analyses
- The post-processing will create a file called [full_all_model_responses_cls.csv](model_responses/full_all_model_responses_cls.csv). It is also available directly with gitlfs for ease of use. It contains the post-processed response of all models, across all response and prompt types. This file can be used with [tableau_analysis.twb](tableau_analysis.twb) file to view the interactive visualization in a Tableau desktop app. The file will have to renamed to `all_model_responses.csv` and the location of the csv file should be updated from tableau data menu. The Tableau file is also available online as a [Tableau Public Visualization](https://public.tableau.com/app/profile/aisha.khatun/viz/LLMReliabilityandConsistencyAnalysis/Sheet1a).
- [response_post_processing.ipynb](experiments/response_post_processing.ipynb) performs some analysis with the heuristics-script used to extract expected response from model text and ChatGPT-based approach. The script works much better with 95\% accurate extraction.
- [response_analysis.ipynb](experiments/response_analysis.ipynb) analyzes the Accuracy, Consistenct, and Resolution of all models. [fulltext_analysis.ipynb](experiments/fulltext_analysis.ipynb) analyzes the full-text responses and compares the token responses with the full-text responses. These files form the bulk of the results in the thesis.
- Overall analysis of the models was done in [this sheet](https://docs.google.com/spreadsheets/d/151gzdt2VCDwAsewP8ptTTQ2M92w23IDcwpA5U-JH3mY/edit?pli=1&gid=885690679#gid=885690679). It is also available as an HTML file: `llm_analysis/summary of analysis.html`

## Citation

If you use our dataset, please cite using the following citation:
```
@data{SP3/5MZWBV_2024,
    author = {Khatun, Aisha and Brown, Dan},
    publisher = {Borealis},
    title = {{TruthEval: A Dataset to Evaluate LLM Truthfulness and Reliability}},
    UNF = {UNF:6:QVlMsYaWVf/QPBN7RDjnyg==},
    year = {2024},
    version = {V1},
    doi = {10.5683/SP3/5MZWBV},
    url = {https://doi.org/10.5683/SP3/5MZWBV}
}

@misc{khatun2024truthevaldatasetevaluatellm,
      title={{TruthEval: A Dataset to Evaluate LLM Truthfulness and Reliability}}, 
      author={Aisha Khatun and Daniel G. Brown},
      year={2024},
      eprint={2406.01855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01855}, 
}
```

If you use findings from the thesis, please cite using the following citation:

```
@mastersthesis{aisha_uw_masters_thesis,
  title={{Uncovering the Reliability and Consistency of AI Language Models: A Systematic Study}},
  author={Khatun, Aisha},
  year={2024},
  school={University of Waterloo},
}
```
