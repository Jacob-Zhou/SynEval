from collections import Counter
from functools import lru_cache
import random
from typing import List, Optional, Set, Tuple, Union
import nltk
import re
import copy
import rich

from utils.common import be_verbs, have_verbs, adverbial_functions


class EmptyType():
    '''
        EmptyType
        :param tag: the tag of the empty type
        :param reference_index (int): the reference index of the empty type
    '''
    reEMPTY = re.compile(r'^(?P<tag>[^-]+)(-(?P<trace>\d+))?$')
    description = {
        "*T*": "trace_of_a_bar_movement",
        "*": "trace_of_a_movement",
        "0": "null_complementizer",
        "*U*": "unit",
        "*?*": "placeholder",
        "*NOT*": "anti-placeholder",
        "*RNR*": "right_node_raising",
        "*ICH*": "interpret_here",
        "*EXP*": "expletive",
        "*PPA*": "permanent_predictable_ambiguity"
    }

    def __init__(self, type: str):
        # *T*-1
        match = self.reEMPTY.match(type)
        if match is None:
            raise ValueError('Invalid empty type: {}'.format(type))
        self.tag = match.group('tag')
        self.reference_index = match.group('trace')

    def __str__(self):
        label = self.tag
        if self.reference_index is not None:
            label += '-' + str(self.reference_index)
        return label

    def __eq__(self, other):
        # ignore trace
        if isinstance(other, str):
            return self == EmptyType(other)
        return self.tag == other.tag

    def get_text(self, remove_numbers=False):
        output_label = copy.deepcopy(self)
        if remove_numbers:
            output_label.reference_index = None
        return str(output_label)

    def get_description(self):
        return self.description.get(self.tag, 'unrecognized')

    def get_mark(self):
        if self.reference_index is not None:
            return f"<E:{self.get_description()}:{self.reference_index}>"
        else:
            return f"<E:{self.get_description()}>"


class Label():
    '''
        Bracket label
        :param tag: the basic tag of the bracket
        :param function (Set): the function of the bracket
        :param trace (int): the identity index of the bracket
        :param coindex (int): the reference index of the bracket
    '''
    reLABEL = re.compile(
        r'^(?P<tag>[^0-9]+)(=(?P<coindex>\d+))?(-(?P<trace>\d+))?$')

    def __init__(self, label: str):
        # NP-SBJ=1-3 or -NONE-
        if label == '-NONE-':
            self.tag = '-NONE-'
            self.function = set()
            self.function_orginal_order = []
            self.trace = None
            self.coindex = None
        else:
            match = self.reLABEL.match(label)
            if match is None:
                raise ValueError('Invalid label: {}'.format(label))
            tag = match.group('tag').split('-')
            self.tag = tag[0]
            self.function = set(tag[1:])
            self.function_orginal_order = tag[1:]
            self.trace = match.group('trace')
            self.coindex = match.group('coindex')

    @property
    def identity_index(self):
        return self.trace

    @property
    def reference_index(self):
        return self.coindex

    def __str__(self):
        label = self.tag
        if len(self.function_orginal_order) > 0:
            label += '-' + '-'.join(self.function_orginal_order)
        if self.coindex is not None:
            label += '=' + str(self.coindex)
        if self.trace is not None:
            label += '-' + str(self.trace)
        return label

    def __eq__(self, other):
        # ignore trace
        if isinstance(other, str):
            return self == Label(other)
        return self.tag == other.tag and self.function == other.function

    def get_text(self,
                 remove_numbers=False,
                 remove_functions=False,
                 sort_functions=False):
        output_label = copy.deepcopy(self)
        if remove_numbers:
            output_label.coindex = None
            output_label.trace = None
        if remove_functions:
            output_label.function = set()
            output_label.function_orginal_order = []
        if sort_functions:
            output_label.function_orginal_order = list(
                sorted(output_label.function_orginal_order))
        return str(output_label)

    def in_set(self,
               tag: Optional[Union[str, Set[str]]] = None,
               function: Optional[Union[str, Set[str]]] = None,
               coindex: Optional[int] = None,
               trace: Optional[int] = None):
        '''
            Check if the bracket matches the given tag, function and trace
            :param tag (Optional[Union[str, Set[str]]]): the tag to match
            :param function (Optional[Union[str, Set[str]]]): the function to match
            :param trace (Optional[int]): the trace to match
        '''
        if tag is not None:
            if isinstance(tag, str):
                tag = {tag}
            if self.tag not in tag:
                return False
        if function is not None:
            if isinstance(function, str):
                function = {function}
            if not function.issubset(self.function):
                return False
        if trace is not None:
            if self.trace != trace:
                return False
        if coindex is not None:
            if self.coindex != coindex:
                return False
        return True


class LabelSet():

    def __init__(self,
                 tag_set: Optional[Union[str, Set[str]]] = None,
                 function_set: Optional[Union[str, Set[str]]] = None,
                 coindex_set: Optional[int] = None,
                 trace_set: Optional[int] = None):
        self.tag_set = tag_set
        self.function_set = function_set
        self.coindex_set = coindex_set
        self.trace_set = trace_set

    def match(self, label: Union[str, Label]):
        if isinstance(label, str):
            label = Label(label)
        return label.in_set(tag=self.tag_set,
                            function=self.function_set,
                            coindex=self.coindex_set,
                            trace=self.trace_set)


def get_tree_id(id: str, positions: List[int]):
    return f"<{id}>-<{'.'.join([str(p) for p in positions if p is not None])}>"


def get_text_from_tree(tree: nltk.Tree, normalize=True, keep_reference=True):
    if normalize:
        if keep_reference:
            words = [
                x[0] if x[1] != '-NONE-' else f"-NONE-:{x[0]}"
                for x in tree.pos()
            ]
        else:
            words = [x[0] for x in tree.pos() if x[1] != '-NONE-']
        text = normalize_words(words)
    else:
        # always keep the reference
        text = ' '.join(tree.leaves())
    return text


def get_plain_text_from_tree(tree: nltk.Tree):
    return get_text_from_tree(tree, normalize=True, keep_reference=False)


def normalize_words(words):
    normalized_words = [normalize_word(word) for word in words]
    text = " ".join([word for word in normalized_words if word != ''])
    # remove the extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove spaces before punctuations
    text = re.sub(r'\s([,.?!:;])', r'\1', text)
    # remove spaces between brackets
    text = re.sub(r'([({\[])\s', r'\1', text)
    text = re.sub(r'\s([)}\]])', r'\1', text)
    # remove spaces between quotes
    text = re.sub(r'([‘“])\s', r'\1', text)
    text = re.sub(r'\s([\'”])', r'\1', text)
    # normalize the quotes
    text = re.sub(r'‘', r"'", text)
    text = re.sub(r'’', r"'", text)
    text = re.sub(r'“', r'"', text)
    text = re.sub(r'”', r'"', text)
    return text


def normalize_word(word):
    """
    Normalize the words in the sentence
    """
    mapping = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LCB-': '{',
        '-RCB-': '}',
        '-LSB-': '[',
        '-RSB-': ']',
        '`': "'‘",
        '``': '“',
        '\'\'': '”'
    }
    if '-NONE-:' in word:
        empty_type = EmptyType(word.split(':')[-1])
        return empty_type.get_mark()
    else:
        return mapping.get(word, word)


# auxiliary classify functions
def is_declarative_sentence(x: List[nltk.Tree]):
    '''
        Check if the tree is a declarative sentence
        :param tree: a nltk.Tree object
    '''
    parent_label = Label(x[-1].label())
    # parent is direct child of ROOT or in a coordination structure
    if parent_label.tag != 'S':
        return False
    if len(x) > 3:
        return False
    if len(x) == 3:
        uncle_labels = {Label(uncle.label()).tag for uncle in x[-2]}
        if 'VP' in uncle_labels or 'NP' in uncle_labels:
            return False
    return True


def is_it_cleft_sentence(x: nltk.Tree):
    '''
        Check if the tree is a it-cleft sentence
        :param tree: a nltk.Tree object
    '''
    return 'CLF' in Label(x.label()).function


def is_passive_sentence(x: nltk.Tree):
    '''
        Check if the tree is a passive sentence
        :param tree: a nltk.Tree object
    '''
    child_labels = [Label(child.label()).tag for child in x]
    if 'VP' not in child_labels:
        return False
    vp_index = child_labels.index('VP')
    vp_types = get_verb_type_from_tree(x[vp_index])
    return any([vp_type['voice'] == 'passive' for vp_type in vp_types])


def has_elder_noun_sibling(x: nltk.Tree):
    elder_noun_siblings = [(i, child) for i, child in enumerate(x)
                           if Label(child.label()).tag == 'NP']
    if len(elder_noun_siblings) == 0:
        return False
    if any([
            get_noun_phrase_type(child)['type'] == 'possessive'
            for _, child in elder_noun_siblings
    ]):
        if len(elder_noun_siblings) == 1:
            return False
        else:
            return True
    else:
        return True


def is_verb_modifier(x: nltk.Tree):
    node_label = Label(x.label())
    if 'PRD' not in node_label.function:
        if not adverbial_functions.isdisjoint(node_label.function):
            # It is a VP adjunct if it has a adverbial function mark
            if 'BNF' in node_label.function and node_label.tag == 'NP':
                # The `NP-BNF` should be considered as a indirect object
                # (S (NP-SBJ I) (VP baked (NP-BNF Doug) (NP a cake)))
                return False
            return True
        elif node_label.tag in {'ADVP', 'PP', 'RB', 'PRT'}:
            return True
    return False


def tree_list_to_str(tree_list: List[nltk.Tree],
                     remove_numbers=True,
                     remove_functions=True):
    '''
        Convert a list of trees to a string
        :param tree_list: a list of nltk.Tree objects
    '''
    return ' '.join([
        Label(tree.label()).get_text(remove_numbers=remove_numbers,
                                     remove_functions=remove_functions,
                                     sort_functions=True) for tree in tree_list
    ])


def clean_pos(pos: Tuple[str, str]):
    # remove the trace and coindex from the word
    if pos[1] != '-NONE-':
        return pos
    else:
        match = re.match(
                r'^(?P<empty>[^=-]+)(=(?P<coindex>\d+))?(-(?P<trace>\d+))?$',
                pos[0])
        assert match is not None
        return (match.group('empty'), pos[1])


def get_text_from_verb_sequence(verb_sequence: List[Tuple[Tuple[str, str],
                                                          List[int],
                                                          nltk.Tree]],
                                keep_adverb=False):
    '''
        Get the text from a verb sequence
        :param verb_sequence: a list of tuples, each tuple contains a word, its position and the tree
    '''
    verbs = []
    for verb_word, position, tree in verb_sequence:
        if verb_word is not None and clean_pos(verb_word)[1] != '-NONE-':
            verbs.append(verb_word[0])
            if position[-1] < len(tree) - 1:
                if keep_adverb:
                    if (Label(tree[position[-1] + 1].label()).tag == 'ADVP'):
                        # verbs.extend(tree[position[-1] + 1].leaves())
                        verbs.extend([
                            word for word, pos in tree[position[-1] + 1].pos()
                            if Label(pos).tag != '-NONE-'
                        ])
                else:
                    if (Label(tree[position[-1] + 1].label()).tag == 'RB'
                            and tree[position[-1] + 1].leaves()[0]
                            in {'not', "n't"}):
                        verbs.append(tree[position[-1] + 1].leaves()[0])
    return normalize_words(verbs)


def flatten_verb_phrase(tree: nltk.Tree):
    '''
        Flatten a verb phrase
        :param tree: a nltk.Tree object
    '''
    verb_sequences = []

    def _flatten_verb_phrase(tree: nltk.Tree, verb_sequence: List,
                             parent_index: List[int]):
        verb_phrase_children = [(i, p) for i, p in enumerate(tree)
                                if Label(p.label()).tag == 'VP']
        verb_words = [(i, p) for i, p in enumerate(tree)
                      if Label(p.label()).tag in
                      {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'TO'}]
        if len(verb_phrase_children) == 0:
            if len(verb_words) > 0:
                for i, p in verb_words:
                    verb_sequences.append(verb_sequence +
                                          [(p.pos()[0], parent_index + [i],
                                            tree)])
            else:
                verb_sequences.append(verb_sequence +
                                      [(None, parent_index + [None], tree)])
        else:
            for vpi, verb_phrase in verb_phrase_children:
                if len(verb_words) == 0:
                    _flatten_verb_phrase(
                        verb_phrase,
                        verb_sequence + [(None, parent_index + [None], tree)],
                        parent_index + [vpi])
                else:
                    for i, p in verb_words:
                        _flatten_verb_phrase(
                            verb_phrase, verb_sequence +
                            [(p.pos()[0], parent_index + [i], tree)],
                            parent_index + [vpi])

    _flatten_verb_phrase(tree, [], [])
    return verb_sequences


def flatten_noun_phrase(tree: nltk.Tree):
    pass


def get_verb_modifier_type(tree: nltk.Tree):
    tree_label = Label(tree.label())
    type = {'type': 'unrecognized'}
    if tree_label.tag == 'ADVP':
        type = {'type': 'adverbial_phrase'}
    elif tree_label.tag == 'PP':
        type = {'type': 'prepositional_phrase'}
    elif tree_label.tag == 'ADJP':
        type = {'type': 'adjective_phrase'}
    elif tree_label.tag == 'NP':
        type = {'type': 'noun_phrase'}
    elif tree_label.tag == 'S':
        type = get_clause_type(tree)
    elif tree_label.tag == 'SBAR':
        type = get_subordinate_clause_type(tree)
    type['function'] = get_adverbial_function(tree_label)
    return type


def get_noun_modifier_type(tree: nltk.Tree):
    tree_label = Label(tree.label())
    type = {'type': 'unrecognized'}
    if tree_label.tag == 'ADVP':
        type = {'type': 'adverbial_phrase'}
    elif tree_label.tag == 'ADJP':
        type = {'type': 'adjective_phrase'}
    elif tree_label.tag == 'PP':
        type = {'type': 'prepositional_phrase'}
    elif tree_label.tag == 'NP':
        type = {
            'type': 'noun_phrase',
            'function': get_adverbial_function(tree_label)
        }
    elif tree_label.tag == 'VP':
        type = get_verb_type_from_tree(tree)
        types = [t['type'] for t in type]
        type = {
            'type': 'reduced_relative_clause',
            'sub_type': ','.join(types),
        }
    elif tree_label.tag == 'RRC':
        type = {'type': 'reduced_relative_clause'}
    elif tree_label.tag == 'S':
        type = get_clause_type(tree)
    elif tree_label.tag == 'SBAR':
        type = get_subordinate_clause_type(tree)
    return type


def get_adverbial_function(label: Label):
    function = label.function
    if 'BNF' in function:
        return 'benefactive'
    elif 'DIR' in function:
        return 'direction'
    elif 'EXT' in function:
        return 'extent'
    elif 'LOC' in function:
        return 'locative'
    elif 'MNR' in function:
        return 'manner'
    elif 'PRP' in function:
        return 'purpose'
    elif 'TMP' in function:
        return 'temporal'
    else:
        return 'no-function'


def get_noun_modifier_function(label: Label):
    function = label.function
    if 'BNF' in function:
        return 'benefactive'
    elif 'DIR' in function:
        return 'direction'
    elif 'EXT' in function:
        return 'extent'
    elif 'LOC' in function:
        return 'locative'
    elif 'MNR' in function:
        return 'manner'
    elif 'PRP' in function:
        return 'purpose'
    elif 'TMP' in function:
        return 'temporal'
    else:
        if label.tag == 'NP':
            if 'ADV' in function:
                return 'per_share_like'
            elif 'TTL' in function:
                return 'title'
        return 'no-function'


def get_subject_type(tree: nltk.Tree):
    subject_tree_label = Label(tree.label())
    if subject_tree_label.tag == 'NP':
        return get_noun_phrase_type(tree)
    elif subject_tree_label.tag == 'S':
        return get_clause_type(tree)
    elif subject_tree_label.tag == 'SBAR':
        return get_subordinate_clause_type(tree)
    elif subject_tree_label.tag == 'PP':
        return {'type': 'prepositional_phrase'}
    elif subject_tree_label.tag == 'ADVP':
        return {'type': 'adverbial_phrase'}
    elif subject_tree_label.tag == 'ADJP':
        return {'type': 'adjective_phrase'}
    else:
        return {
            'type':
            f'unrecognized({subject_tree_label.tag}->{[Label(child.label()).tag if isinstance(child, nltk.Tree) else child for child in tree]})'
        }


def get_noun_phrase_type(tree: nltk.Tree):
    children = [
        Label(child.label()).tag if child.label() != '-NONE-' else '-NONE-'
        for child in tree
    ]
    if children[-1] == 'POS':
        return {'type': 'possessive'}
    else:
        headtag = None
        head_index = None
        for i, child in enumerate(reversed(children)):
            if child in {'NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR'}:
                headtag = child
                head_index = len(children) - i - 1
                break
        if headtag in {'NN', 'NNP', 'NNPS', 'NNS', 'NX'}:
            return {'type': 'noun_phrase'}
        elif headtag == 'JJR':
            return {'type': 'comparative_adjective_phrase'}
        elif headtag == 'POS':
            return {'type': 'possessive'}
        elif not set(children).isdisjoint({'CC', 'CONJP'}):
            return {'type': 'coordinated_noun_phrase'}
        else:
            # search for the first NP child
            find_np = False
            for i, child in enumerate(children):
                if child == 'NP':
                    find_np = True
                    break
            if find_np:
                return get_noun_phrase_type(tree[i])
            else:
                headtag = None
                head_index = None
                # for left-to-right head finding
                for i, child in enumerate(reversed(children)):
                    if child in {'$', 'ADJP', 'PRN'}:
                        headtag = child
                        head_index = len(children) - i - 1
                        break
                if headtag == '$':
                    return {'type': 'currency'}
                elif headtag == 'ADJP':
                    return {'type': 'adjective_phrase'}
                elif headtag == 'PRN':
                    return {'type': 'parenthetical_phrase'}
                else:
                    headtag = None
                    head_index = None
                    # for left-to-right head finding
                    for i, child in enumerate(reversed(children)):
                        if child in {'CD'}:
                            headtag = child
                            head_index = len(children) - i - 1
                            break
                    if headtag == 'CD':
                        return {'type': 'number'}
                    else:
                        headtag = None
                        head_index = None
                        # for left-to-right head finding
                        for i, child in enumerate(reversed(children)):
                            if child in {'JJ', 'JJS', 'RB', 'QP'}:
                                headtag = child
                                head_index = len(children) - i - 1
                                break
                        if headtag in {'JJ', 'JJS'}:
                            return {'type': 'adjective_phrase'}
                        elif headtag == 'RB':
                            return {'type': 'adverbial_phrase'}
                        elif headtag == 'QP':
                            return {'type': 'quantifier_phrase'}
                        else:
                            last_word = children[-1]
                            if last_word == 'PRP':
                                return {'type': 'pronoun'}
                            elif last_word == 'DT':
                                return {'type': 'determiner'}
                            elif last_word == 'CD':
                                return {'type': 'number'}
                            elif last_word == '-NONE-':
                                empty_label = EmptyType(tree.leaves()[-1])
                                if empty_label.tag == '*EXP*':
                                    return {'type': 'it-extraposition'}
                                else:
                                    return {
                                        'type': 'empty',
                                        'empty_type': empty_label.tag,
                                        'trace': empty_label.reference_index
                                    }
                            elif last_word == 'EX':
                                return {'type': 'existential_there'}
                            elif last_word in {'JJ', 'JJR', 'JJS'}:
                                return {'type': 'adjective_phrase'}
                            elif last_word in {'NN', 'NNS', 'NNP', 'NNPS'}:
                                return {'type': 'noun_phrase'}
                            else:
                                return {'type': 'unrecognized'}


def get_clause_type(tree: nltk.Tree):
    clause_label = Label(tree.label())
    if clause_label.tag == 'SBAR':
        return get_subordinate_clause_type(tree)
    if "CLF" in clause_label.function:
        return {'type': 'cleft_clause'}
    elif clause_label.tag == 'SINV':
        return {'type': 'invertion_clause'}
    elif len(tree) == 2:
        first_child_label = Label(tree[0].label())
        if first_child_label.tag == 'NP' and 'SBJ' in first_child_label.function and Label(
                tree[1].label()).tag == 'VP':
            verb_phrase_types = get_verb_type_from_tree(tree[1])
            if len(verb_phrase_types) > 1:
                return {'type': 'unrecognized(coordination_nonfinite)'}
            elif len(verb_phrase_types) == 0:
                return {'type': 'unrecognized(zero_verb)'}
            else:
                verb_phrase_type = verb_phrase_types[0]
                if verb_phrase_type['type'] == 'infinitive':
                    return {'type': 'infinitive_clause'}
                elif verb_phrase_type['type'] == 'present_participle':
                    return {'type': 'participial_clause'}
                else:
                    return {'type': f'normal_clause'}
        if first_child_label.tag == 'NP' and 'SBJ' in first_child_label.function and 'PRD' in Label(
                tree[1].label()).function:
            # small clause
            predicate_label = Label(tree[1].label()).tag
            if predicate_label == 'NP':
                return {'type': 'small_clause(nominal)'}
            elif predicate_label == 'ADJP':
                return {'type': 'small_clause(adjective)'}
            elif predicate_label == 'PP':
                return {'type': 'small_clause(pseudo-adjective)'}
            else:
                return {
                    'type': f'unrecognized(small_clause->{predicate_label})'
                }
        else:
            return {
                'type':
                f'unrecognized({clause_label.tag}->{[Label(child.label()).tag for child in tree]})'
            }
    elif len(tree) == 1 and Label(tree[0].label()).tag == '-NONE-':
        empty_label = EmptyType(tree.pos()[0][0])
        return {
            'type': 'empty_clause',
            'empty_type': empty_label.tag,
            'trace': empty_label.reference_index
        }
    else:
        return {'type': f'normal_clause'}


def get_subordinate_clause_type(tree: nltk.Tree):
    subordinate_label = Label(tree.label())
    first_child_label = Label(tree[0].label())
    if first_child_label.tag.startswith('WH'):
        if clean_pos(tree[0].pos()[0])[0] == '0':
            return {'type': 'wh_clause(null_complementizer)'}
        else:
            return {'type': 'wh_clause'}
    elif first_child_label.tag == 'IN':
        return {'type': 'that_clause'}
    elif first_child_label.tag == '-NONE-' and clean_pos(
            tree[0].pos()[0])[0] == '0':
        return {'type': 'that_clause(null_complementizer)'}
    elif len(tree) == 1 and first_child_label.tag == '-NONE-':
        empty_label = EmptyType(tree.pos()[0][0])
        return {
            'type': 'empty_clause',
            'empty_type': empty_label.tag,
            'trace': empty_label.reference_index
        }
    elif {Label(child.label()).tag
          for child in tree}.issubset({'CC', 'CONJP', 'SBAR', ','}):
        return {'type': 'coordinated_subordinate_clause'}
    else:
        return {
            'type':
            f'unrecognized({subordinate_label.tag}->{[Label(child.label()).tag for child in tree]})'
        }


def get_verb_type_from_verb_sequence(verb_sequence: List):
    tense = [None, None]
    voice = 'active'
    modal_word = None
    is_future = False
    type = None
    if len(verb_sequence) < 1:
        return {
            'type': 'mislabeled',
            'tense': tuple(tense),
            'voice': None,
        }
    elif verb_sequence[-1][0] is None:
        is_referenced_brackets = [
            Label(x.label()).reference_index is not None
            for x in verb_sequence[-1][2]
        ]
        if all(is_referenced_brackets):
            # gapping
            return {
                'type': 'gapping',
                'tense': tuple(tense),
                'voice': None,
            }
        elif any(is_referenced_brackets):
            return {
                'type': 'mislabeled(gapping)',
                'tense': tuple(tense),
                'voice': None,
            }
    # remove all None verb
    verb_sequence = [x for x in verb_sequence if x[0] is not None]
    if len(verb_sequence) <= 0:
        return {
            'type': 'mislabeled',
            'tense': tuple(tense),
            'voice': None,
        }
    if verb_sequence[0][0][1] == 'MD':
        modal_word, _ = verb_sequence[0][0]
        if modal_word in {'will', 'wo', "'ll"}:
            # future tense
            tense[0] = 'future'
            is_future = True
        elif modal_word == 'would':
            # future in the past tense
            tense[0] = 'future_in_past'
            is_future = True
        verb_sequence = verb_sequence[1:]
    if len(verb_sequence) == 0:
        return {
            'type': 'unrecognized',
            'tense': tuple(tense),
            'voice': None,
        }
    if verb_sequence[0][0][1] == 'TO':
        type = 'infinitive'
        verb_sequence = verb_sequence[1:]
    if len(verb_sequence) == 0:
        return {
            'type': 'unrecognized',
            'tense': tuple(tense),
            'voice': None,
        }
    if not (1 <= len(verb_sequence) <= 3):
        return {
            'type': 'unrecognized(verb_sequence_length)',
            'tense': tuple(tense),
            'voice': None,
        }
    # rich.print([item[:3] for item in verb_sequence])
    # the first verb determines the tense
    first_verb_word, first_verb_pos = verb_sequence[0][0]
    if first_verb_pos in {'VBD', 'VBZ', 'VBP', 'VB'}:
        if first_verb_pos == 'VBD':
            type = 'normal'
            tense[0] = 'past'
        elif first_verb_pos in {'VBZ', 'VBP'}:
            type = 'normal'
            tense[0] = 'present'
        elif first_verb_pos == 'VB':
            if type is None:
                if modal_word is None:
                    type = 'bare_infinitive'
                elif is_future:
                    type = 'normal'
                else:
                    type = 'modal'
        if first_verb_word in be_verbs:
            if len(verb_sequence) > 1 and verb_sequence[1][0][1] == 'VBN':
                # was broken
                voice = 'passive'
                if modal_word == 'shall':
                    type = 'normal'
                    is_future = True
                    tense[0] = 'future'
            elif len(verb_sequence) > 1 and verb_sequence[1][0][1] == 'VBG':
                # was breaking
                tense[1] = 'continuous'
                if len(verb_sequence) > 2 and verb_sequence[2][0][1] == 'VBN':
                    # was being broken
                    voice = 'passive'
        elif first_verb_word in have_verbs:
            # had
            if len(verb_sequence) > 1 and verb_sequence[1][0][1] == 'VBN':
                # had broken
                tense[1] = 'perfect'
                if verb_sequence[1][0][0] == 'been':
                    if len(verb_sequence
                           ) > 2 and verb_sequence[2][0][1] == 'VBG':
                        # had been breaking
                        tense[1] += '_continuous'
                        # there is no passive form for past perfect continuous
                    elif len(verb_sequence
                             ) > 2 and verb_sequence[2][0][1] == 'VBN':
                        # had been broken
                        voice = 'passive'
    elif first_verb_pos == 'VBG':
        type = 'present_participle'
        if first_verb_word == 'being' and len(
                verb_sequence) > 1 and verb_sequence[1][0][1] == 'VBN':
            # being broken
            voice = 'passive'
        elif first_verb_word == 'having' and len(
                verb_sequence) > 1 and verb_sequence[1][0][1] == 'VBN':
            # having broken
            type = 'perfect_participle'
            tense[1] = 'perfect'
            if verb_sequence[1][0][0] == 'been' and len(
                    verb_sequence) > 2 and verb_sequence[2][0][1] == 'VBN':
                # having been broken
                voice = 'passive'
    elif first_verb_pos == 'VBN':
        type = 'past_participle'
        voice = 'passive'
    return {
        'type': type,
        'tense': tuple(tense),
        'voice': voice,
    }


def get_verb_type_from_tree(tree: nltk.Tree):
    # find right-most VB
    verb_sequences = flatten_verb_phrase(tree)
    return [
        get_verb_type_from_verb_sequence(verb_sequence)
        for verb_sequence in verb_sequences
    ]


def search_bracket(tree: Union[nltk.Tree, List[nltk.Tree]],
                   label: Optional[Union[str, Label, LabelSet]] = None,
                   self_filter_fn: Optional[callable] = None,
                   parent_filter_fn: Optional[callable] = None,
                   elder_siblings_filter_fn: Optional[callable] = None,
                   younger_siblings_filter_fn: Optional[callable] = None,
                   children_filter_fn: Optional[callable] = None,
                   recursive: bool = True):
    '''
        Search for a bracket with the given label in the tree
        :param tree: a nltk.Tree object
        :param tag (Optional[Union[str, Set[str]]]): the tag to match
        :param function (Optional[Union[str, Set[str]]]): the function to match
        :param coindex (Optional[int]): the coindex to match
        :param trace (Optional[int]): the trace to match
        :param parent_filter_fn (Optional[str]): the regex to match the parent label
        :param elder_siblings_filter_fn (Optional[str]): the regex to match the elder siblings label
        :param younger_siblings_filter_fn (Optional[str]): the regex to match the younger siblings label
        :param children_filter_fn (Optional[str]): the regex to match the children label

        :return: a list of dict, each dict contains the matched bracket and its parent, elder siblings, younger siblings and children
        
    '''
    matched_bracket = []

    def _search_bracket(tree: nltk.Tree, parent: List[nltk.Tree],
                        elder_siblings: List[nltk.Tree],
                        younger_siblings: List[nltk.Tree],
                        positions: List[int]):
        if not isinstance(tree, nltk.Tree):
            return
        if label is None:
            tree_match = True
        elif isinstance(label, str):
            tree_match = Label(tree.label()) == Label(label)
        elif isinstance(label, Label):
            tree_match = Label(tree.label()) == label
        elif isinstance(label, LabelSet):
            tree_match = label.match(tree.label())
        elif callable(label):
            tree_match = label(Label(tree.label()))

        if tree_match and self_filter_fn is not None:
            tree_match = self_filter_fn(tree)

        if tree_match and parent_filter_fn is not None:
            tree_match = parent_filter_fn(parent)

        if tree_match and elder_siblings_filter_fn is not None:
            tree_match = elder_siblings_filter_fn(elder_siblings)

        if tree_match and younger_siblings_filter_fn is not None:
            tree_match = younger_siblings_filter_fn(younger_siblings)

        if tree_match and children_filter_fn is not None:
            tree_match = children_filter_fn(tree)

        if tree_match:
            matched_bracket.append({
                'tree': tree,
                'positions': positions,
            })
        if not recursive:
            return
        for i, child in enumerate(tree):
            _search_bracket(child, parent + [tree], tree[:i], tree[i + 1:],
                            positions + [i])

    if isinstance(tree, nltk.Tree):
        trees = [tree]
    elif isinstance(tree, list):
        trees = tree
    else:
        raise ValueError('Invalid tree type: {}'.format(type(tree)))

    for i, tree in enumerate(trees):
        _search_bracket(tree, [], [], [], [i])
    return matched_bracket


# get the matrix of LCS lengths at each sub-step of the recursive process
# (m+1 by n+1, where m=len(list1) & n=len(list2) ... it's one larger in each direction
# so we don't have to special-case the x-1 cases at the first elements of the iteration
def lcs_mat(list1, list2):
    m = len(list1)
    n = len(list2)
    # construct the matrix, of all zeroes
    mat = [[0] * (n + 1) for row in range(m + 1)]
    # populate the matrix, iteratively
    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if list1[row - 1] == list2[col - 1]:
                # if it's the same element, it's one longer than the LCS of the truncated lists
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                # they're not the same, so it's the the maximum of the lengths of the LCSs of the two options (different list truncated in each case)
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])
    # the matrix is complete
    return mat


# backtracks all the LCSs through a provided matrix
def all_lcs(lcs_dict, mat, list1, list2, index1, index2):
    # if we've calculated it already, just return that
    if ((index1, index2) in lcs_dict): return lcs_dict[(index1, index2)]
    # otherwise, calculate it recursively
    if (index1 == 0) or (index2 == 0):  # base case
        return [[]]
    elif list1[index1 - 1] == list2[index2 - 1]:
        # elements are equal! Add it to all LCSs that pass through these indices
        lcs_dict[(index1, index2)] = [
            prevs + [(index1 - 1, index2 - 1, list1[index1 - 1])]
            for prevs in all_lcs(lcs_dict, mat, list1, list2, index1 -
                                 1, index2 - 1)
        ]
        return lcs_dict[(index1, index2)]
    else:
        lcs_list = []  # set of sets of LCSs from here
        # not the same, so follow longer path recursively
        if mat[index1][index2 - 1] >= mat[index1 - 1][index2]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1, index2 - 1)
            for series in before:  # iterate through all those before
                if not series in lcs_list:
                    lcs_list.append(
                        series
                    )  # and if it's not already been found, append to lcs_list
        if mat[index1 - 1][index2] >= mat[index1][index2 - 1]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1 - 1, index2)
            for series in before:
                if not series in lcs_list: lcs_list.append(series)
        lcs_dict[(index1, index2)] = lcs_list
        return lcs_list


# return a set of the sets of longest common subsequences in list1 and list2
def lcs(list1, list2):
    # mapping of indices to list of LCSs, so we can cut down recursive calls enormously
    mapping = dict()
    # start the process...
    return all_lcs(mapping, lcs_mat(list1, list2), list1, list2, len(list1),
                   len(list2))


def upper_first_letter(s):
    return s[0].upper() + s[1:]


if __name__ == '__main__':
    tree1 = nltk.Tree.fromstring(
        "(VP (VBP file) (NP (PRP$ their) (NNS reports)) (ADVP-TMP (RB late)))")
    tree_type1 = get_verb_type_from_tree(tree1)