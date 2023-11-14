<div align="center">

# How Well Do Large Language Models Understand Syntax? An Evaluation by Asking Natural Language Questions
__Houquan Zhou__, Yang Hou, Zhenghua Li, Xuebin Wang, Zhefeng Wang, Xinyu Duan, Min Zhang

</div>

## TL;DR
This repository contains the code and data for the paper [How Well Do Large Language Models Understand Syntax? An Evaluation by Asking Natural Language Questions](https://arxiv.org).

## Setup

Clone this repo:
```sh
git clone https://github.com/Jacob-Zhou/SynEval.git
```

Then you can use following commands to create an environment and install the dependencies:
```sh
. scripts/set_environment.sh
```

To generate the data for the experiments, you should first download the [Penn Treebank 3](https://catalog.ldc.upenn.edu/LDC99T42) datasets and put the `LDC1999T42/PARSED/MRG/WSJ` folder into the `data` folder.
The `data` folder should have the following structure:
```sh
data
└── WSJ
    ├── 00
    │   ├── WSJ_0000.MRG
    │   ├── WSJ_0001.MRG
    ...
    └── 24
        ├── WSJ_2400.MRG
        ├── WSJ_2401.MRG
        ...
        └── WSJ_2454.MRG
```

Then you can run the following command to generate the data for the experiments:
```sh
bash build.sh
```

After that, generated data will be in the `generated` folder.
The question will be in the json format.
The format of the question is as follows:
```json
{
    "question": "In the above sentence, the grammatical subject of `am` is ____________.",
    "answer": "I",
    "type": "fill_in_the_blank",
    "tags": [
        "subtype:normal_clause",
        "surface_subject:pronoun",
        "surface_subject:voice:active"
    ],
    "source": [
        "<XX>-<Y>",
        "<XX>-<Y.Z>"
    ],
    "question_template": "In the above sentence, the grammatical subject of `{verb_phrase}` is ____________.",
    "md5": "...",
    "knowledge_point": "surface_subject",
    "sentence": "I am Batman.",
    "generation_method": "generate_surface_subject_question_by_verb_phrase",
    "id": xxxxxxx
}
```

## Run

We provide the scripts to run the experiments.
You can run the following command to evaluate the model:

```sh
python test.py \ 
    --model-name gpt-3.5-turbo \    # > (Required) The name of the model to use.
    --suite v1.0 \                  # > (Required) Name of the suite this run belongs to.
    --task-name syneval \           #   (Optional) The evaluation task.
    --model-revision main \         #   (Optional) The revision of the model to use.
    --lora-weights None \           #   (Optional) The path to the lora weights.
    --bit-8 \                       #   (Optional) Use 8-bit precision.
    --sample-size 5 \               #   (Optional) The sample size for test set.
    --n-exemplar 5 \                #   (Optional) The number of exemplars to use.
    --seed 42 \                     #   (Optional) Random seed.
    --n-workers None \              #   (Optional) Number of multiprocessing workers.
    --force \                       #   (Optional) Force re-evaluation if results already exist.
    --ignore-code-revision \        #   (Optional) Ignore code revision.
    --save-results-per 25 \         #   (Optional) Save results per x evalation.
    --exemplar-type all \           #   (Optional) The type of exemplars to use.
    --debug-mode \                  #   (Optional) Run in debug mode.
    --max-eval-instances None \     #   (Optional) Maximum number of instances to evaluate on.
    --dry-run                       #   (Optional) Dry run, do not actually request model.
```

We also provide the scripts to run the experiments all at once.

__NOTE__: Some of the following scripts will generate slurm scripts and submit them to the cluster.
### Random
```sh
bash run_random.sh
```

### ChatGPT
```sh
bash run_gpt.sh
```

### HuggingFace Models
```sh
bash run_all.sh
```

### Case Study on BaiChuan
```sh
bash run_baichuan_revision.sh
```
If you meet the error `AttributeError: 'BaiChuanTokenizer' object has no attribute 'sp_model'`, please refer to the [IMPORTANT NOTE !!!](#IMPORTANT-NOTE-!!!) section.

## IMPORTANT NOTE !!!
In the Transformers library, version 4.34.0.dev0 or later, an error may be raised if you use `Baichuan` or `ChatGLM`. 
To resolve this, you should modify the tokenization_baichuan.py and tokenization_chatglm.py files. 
Move the `super().__init__(*)` call in each file to a point after the initialization of `self.tokenizer` in tokenization_baichuan.py and `self.sp_model` in tokenization_chatglm.py, respectively.
Otherwise, you can downgrade the transformers library to version 4.33.0.
However, this version does not support `mistralai/Mistral-7B-v0.1` and `mistralai/Mistral-7B-Instruct-v0.1`.