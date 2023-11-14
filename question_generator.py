from datetime import datetime
import ujson as json
import argparse
import glob
import os
import nltk
import rich
from tqdm import tqdm
from utils.extraction import ExtractionLoader
from multiprocessing import Pool

debug_mode = False
treebank_info_file = 'extracted/results-*.json'
treebank_pattern = 'data/WSJ/*/*.MRG'

global_question_id = 0

if debug_mode:
    rich.print('[bold red]Debug mode is on[/bold red]')
    treebank_info_file = 'extracted/debug.json'


def get_treebank_infos():
    # get the newest file
    json_file = sorted(glob.glob(treebank_info_file), key=os.path.getmtime)[-1]
    rich.print(f"[bold green]Loading {json_file}[/bold green]")
    treebank_infos = json.load(open(json_file, 'r'))

    return treebank_infos


def get_trees():
    # 1. read in all trees
    trees = []
    for file in sorted(glob.glob(treebank_pattern)):
        tree_buffer = ''
        with open(file) as f:
            for line in f:
                if line.startswith('('):
                    if tree_buffer.strip() != '':
                        trees.append(tree_buffer.strip())
                    tree_buffer = line.strip() + ' '
                else:
                    tree_buffer += line.strip() + ' '
            else:
                if tree_buffer.strip() != '':
                    trees.append(tree_buffer.strip())
    return trees


def merge_generations(generations):
    global global_question_id
    instances = []
    for extraction in generations:
        instances.extend(extraction['instances'])
    for instance in instances:
        instance['id'] = global_question_id
        global_question_id += 1
    return instances


class GenerationWrapper():

    def __init__(self, generation_suites):
        self.generation_suites = generation_suites

    def __call__(self, x):
        return self.generation_suites(tree=nltk.Tree.fromstring(x[0]),
                                      json_obj=x[1])


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # get all trees
    trees = get_trees()
    treebank_infos = get_treebank_infos()
    loader = ExtractionLoader()
    loader.extract_method_prefix = 'generate'
    if debug_mode:
        generation_suites = loader.discover(start_dir='question_generators',
                                            pattern='generate_*.py')
    else:
        generation_suites = loader.discover(start_dir='question_generators',
                                            pattern='generate_*.py')
    instances = []
    if debug_mode:
        rich.print(generation_suites)
        for tree, treebank_info in tqdm(zip(trees, treebank_infos),
                                        total=len(trees)):
            tree = nltk.Tree.fromstring(tree)
            result = generation_suites(tree=tree, json_obj=treebank_info)
            instances.extend(merge_generations(result))
            rich.print(result)
    else:
        # multi-processing
        saved = False
        with Pool(8) as p:
            for result in tqdm(p.imap(GenerationWrapper(generation_suites),
                                      zip(trees, treebank_infos),
                                      chunksize=100),
                               total=len(trees)):
                instances.extend(merge_generations(result))
                if not saved and len(instances) >= 25:
                    with open(os.path.join(args.output_dir, f"debug.json"),
                              'w') as f:
                        json.dump(instances, f, indent=4)
                    saved = True
    if not debug_mode:
        with open(
                os.path.join(
                    args.output_dir,
                    f"results-{datetime.now().strftime('%Y%m%d%H%M%S')}.json"),
                'w') as f:
            json.dump(instances, f, indent=4)
    else:
        with open(os.path.join(args.output_dir, f"debug.json"), 'w') as f:
            json.dump(instances, f, indent=4)
    rich.print(
        f"[bold green]Total number of instances: {len(instances)}[/bold green]"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()

    main(args)