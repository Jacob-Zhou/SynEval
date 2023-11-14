from collections import Counter, defaultdict
from datetime import datetime
import ujson as json
import argparse
import glob
import os
from typing import List, Union
import nltk
import rich
from tqdm import tqdm
from utils.extraction import ExtractionLoader
from multiprocessing import Pool

from utils.fn import Label, get_plain_text_from_tree, get_text_from_tree

debug_mode = False
treebank_pattern = 'data/WSJ/*/*.MRG'

if debug_mode:
    rich.print('[bold red]Debug mode is on[/bold red]')
    treebank_pattern = 'data/ptb/debug.pid'


def merge_extractions(extractions, tree: Union[List[str], nltk.Tree]):
    results = defaultdict(dict)
    for extraction in extractions:
        for extraction_result in extraction['results']:
            for k, v in extraction_result['result'].items():
                if k not in results[extraction_result['tree_id']]:
                    results[extraction_result['tree_id']][k] = v
                else:
                    if isinstance(v, dict):
                        results[extraction_result['tree_id']][k].update(v)
                    elif isinstance(v, list):
                        results[extraction_result['tree_id']][k].extend(v)
                    elif isinstance(v, Counter):
                        results[extraction_result['tree_id']][k].update(v)
                    else:
                        results[extraction_result['tree_id']][k] = v
    # sort by tree_id
    results = dict(
        sorted(results.items(), key=lambda x: x[0].split('-')[1][1:-1]))
    for k, v in results.items():
        # sort by key, but not recursively
        # first: type, text, others..., (modifiers, appositives if any)
        try:
            type = v['type']
            text = v['text']
            plain_text = v['plain_text']
            v.pop('type', None)
            v.pop('text', None)
            v.pop('plain_text', None)
        except KeyError:
            tree_id = int(k.split('-')[0][1:-1])
            position = k.split('-')[-1][1:-1]
            if position == '':
                if isinstance(tree, nltk.Tree):
                    current_tree = tree
                else:
                    current_tree = nltk.Tree.fromstring(tree[tree_id])
            else:
                position = list(map(int, k.split('-')[-1][1:-1].split('.')))
                if isinstance(tree, nltk.Tree):
                    current_tree = tree[position]
                else:
                    current_tree = nltk.Tree.fromstring(
                        tree[tree_id])[position]
            current_tree_label = current_tree.label()
            if current_tree_label == '':
                current_tree_label = 'ROOT'
            type = f"unrecognized({Label(current_tree_label).get_text(remove_numbers=True)})"
            text = get_text_from_tree(current_tree)
            plain_text = get_plain_text_from_tree(current_tree)
        modifiers = v.get('modifiers', [])
        appositives = v.get('appositives', [])
        v.pop('modifiers', None)
        v.pop('appositives', None)
        results[k] = {
            "type": type,
            "text": text,
            "plain_text": plain_text,
            **dict(sorted(v.items(), key=lambda x: x[0])),
        }
        if len(modifiers) > 0:
            results[k]['modifiers'] = modifiers
        if len(appositives) > 0:
            results[k]['appositives'] = appositives

    return results


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


class ExtractionWrapper():

    def __init__(self, extraction_suites):
        self.extraction_suites = extraction_suites

    def __call__(self, x):
        return self.extraction_suites(tree=nltk.Tree.fromstring(x[1]),
                                      tree_id=str(x[0]))


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # get all trees
    trees = get_trees()
    loader = ExtractionLoader()
    if debug_mode:
        loader.extract_method_prefix = 'extract_noun_postmodifier'
        extraction_suites = loader.discover(start_dir='extractions')
    else:
        extraction_suites = loader.discover(start_dir='extractions')
    results = []
    if debug_mode:
        rich.print(extraction_suites)
        for i, tree in tqdm(enumerate(trees), total=len(trees)):
            # if i > 10:
            #     break
            tree = nltk.Tree.fromstring(tree)
            result = extraction_suites(tree=tree, tree_id=str(i))
            results.append(merge_extractions(result, tree))
            tree.pretty_print(unicodelines=True)
            rich.print(result)
    else:
        # multi-processing
        with Pool(8) as p:
            for result in tqdm(p.imap(ExtractionWrapper(extraction_suites),
                                      enumerate(trees),
                                      chunksize=250),
                               total=len(trees)):
                results.append(merge_extractions(result, trees))
                if len(results) == 25:
                    with open(os.path.join(args.output_dir, f"debug.json"),
                              'w') as f:
                        json.dump(results, f, indent=4)
    if not debug_mode:
        with open(
                os.path.join(
                    args.output_dir,
                    f"results-{datetime.now().strftime('%Y%m%d%H%M%S')}.json"),
                'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()

    main(args)