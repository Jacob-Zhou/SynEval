# a dummy file to status the type in json file
# Path: dummy.py

import glob
import random
import ujson as json
import os
import rich
from collections import Counter, defaultdict
from rich.table import Table
from tqdm import tqdm

debug_mode = False
question_info_file = 'generated/results-*.json'

global_question_id = 0

random.seed(42)

if debug_mode:
    rich.print('[bold red]Debug mode is on[/bold red]')
    question_info_file = 'generated/debug.json'


def get_question_infos():
    # get the newest file
    json_file = sorted(glob.glob(question_info_file), key=os.path.getmtime)[-1]
    rich.print(f"[bold green]Loading {json_file}[/bold green]")
    treebank_infos = json.load(open(json_file, 'r'))

    return treebank_infos


questions = get_question_infos()
tags = Counter()
question_types = Counter()
question_dict = defaultdict(list)
question_type_dict = defaultdict(list)
for question in tqdm(questions):
    question_tags = [
        tag.replace('(non-restrictive)', '') for tag in question['tags']
    ]
    tags.update([" ".join(question_tags)])
    question_types.update([question['type']])
    try:
        for tag in question_tags:
            question_dict[
                f"{question['knowledge_point']}@{question['type']}@{tag}"].append(
                    question)
    except KeyError:
        rich.print(question)
        raise Exception(f"modifier_info does not have type")
    question_type_dict[question['type']].append(question)

# show them in a descending order in a pretty way

table = Table(title="Question Types", show_lines=True)
table.add_column("Type", justify="left", style="cyan", min_width=28)
table.add_column("Count", justify="right", style="black")
for k, v in question_types.most_common():
    table.add_row(k, f"{v}")
rich.print(table)

sample_sizes = [5]
sampled_questions = defaultdict(list)

sampled_question_ids = set()
for question_type, question_list in question_dict.items():
    if len(question_list) < 50:
        continue
    random_question_list = random.sample(
        question_list, min(len(question_list), max(sample_sizes)))
    for question in random_question_list:
        sampled_question_ids.add(question['id'])

    for sample_size in sample_sizes:
        sampled_questions[sample_size].extend(
            random_question_list[:sample_size])
    rich.print(
        f"[bold green]Randomly sampling {sample_size} questions from {len(question_list)} questions of type {question_type}[/bold green]"
    )

# sample exemplar, make sure they are not in the sampled questions
exemplars = defaultdict(list)
for question_type, question_list in question_dict.items():
    type = question_type.split('@')[1]
    if len(question_list) < 50:
        continue
    question_list = [
        question for question in question_list
        if question['id'] not in sampled_question_ids
    ]
    question_list = random.sample(question_list, min(len(question_list), 50))
    for question in question_list:
        if question['id'] not in sampled_question_ids:
            exemplars[type].append(question)

for type in exemplars:
    exemplars[type] = random.sample(exemplars[type],
                                    min(50, len(exemplars[type])))

syntactic_exemplars = defaultdict(list)
for question_type, question_list in question_dict.items():
    if len(question_list) < 50:
        continue
    question_list = random.sample(question_list, min(len(question_list), 50))
    for question in question_list:
        if question['id'] not in sampled_question_ids:
            type = question['type']
            syntactic_knowledge_point = question['knowledge_point']
            syntactic_exemplars[f"{syntactic_knowledge_point}@{type}"].append(
                question)

for type in syntactic_exemplars:
    syntactic_exemplars[type] = random.sample(
        syntactic_exemplars[type], min(50, len(syntactic_exemplars[type])))

table = Table(title="Syntactic Exemplars", show_lines=True)
table.add_column("Type", justify="left", style="cyan", min_width=28)
table.add_column("Count", justify="right", style="black")
for k, v in syntactic_exemplars.items():
    table.add_row(k, f"{len(v)}")
rich.print(table)

rich.print(
    f"[bold green]Randomly sampled {len(sampled_questions)} questions[/bold green]"
)
json.dump(list(sorted(sampled_question_ids)),
          open('generated/test.random-sample.ids.json', 'w'),
          indent=4)
json.dump(syntactic_exemplars,
          open('generated/test.syntactic-exemplars.json', 'w'),
          indent=4)
json.dump(exemplars, open('generated/test.exemplars.json', 'w'), indent=4)

for sample_size in sample_sizes:
    json.dump(sampled_questions[sample_size],
              open(f'generated/test.random-sample-{sample_size}.json', 'w'),
              indent=4)
json.dump(list(sorted(sampled_question_ids)),
          open('generated/test.random-sample.ids.json', 'w'),
          indent=4)
