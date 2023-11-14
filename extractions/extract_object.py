from collections import Counter
import copy
import nltk
import rich
from utils.extraction import TreeExtraction
from utils.fn import (Label, get_plain_text_from_tree, get_text_from_tree,
                      get_verb_type_from_verb_sequence, flatten_verb_phrase,
                      get_subject_type, is_declarative_sentence,
                      is_it_cleft_sentence, is_passive_sentence,
                      is_verb_modifier, search_bracket, clean_pos, get_tree_id)

# 1.1 sentence-level
# 1.1.1 subject in declarative sentence (active)
# 1.1.2 subject in declarative sentence (passive) [have a surface subject and possibly a logical subject]
# 1.1.3 subject in interrogative sentence (TODO)
# 1.1.4 subject in imperative sentence (implicit subject)
# 1.1.5 subject in topicalized sentence [the subject is moved to the front of the sentence]
# 1.1.6 subject in it-cleft sentence
# 1.1.7 subject in it-extraposition sentence
# 1.1.8 subject in there-extraposition sentence

# 1.2 clause-level in which the subject sometime is moved to outside the clause
# 1.2.1 subject in relative clause
# 1.2.2 subject in infinitive clause
# 1.2.3 subject in participial clause


class ObjectExtraction(TreeExtraction):

    def extract_object_in_declarative_clause(self, tree_id: str,
                                             tree: nltk.Tree):
        label_fn = lambda x: x.tag == 'VP'

        # highest VP
        parent_filter_fn = (lambda x: Label(x[-1].label()).tag != 'VP')

        candidate_brackets = search_bracket(tree,
                                            label=label_fn,
                                            parent_filter_fn=parent_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            # The candidate bracket is the highest VP
            verb_sequences = flatten_verb_phrase(candidate_bracket['tree'])
            for verb_sequence in verb_sequences:
                object = None
                indirect_object = None
                complement = None
                subject_complement = None
                # object must be attached to the last verb
                # *-PRD or NP, S or SBAR, indirect object must be a NP
                verb_word, position, *_ = verb_sequence[-1]
                if verb_word is None:
                    # gapping VP
                    continue
                predicate_absolute_positions = candidate_bracket['positions'][
                    1:] + position
                parent_positons = candidate_bracket['positions'][
                    1:] + position[:-1]
                parent_tree = tree[parent_positons]
                search_space = parent_tree[position[-1] + 1:]
                # check if there is a ':' in the search space
                if any([Label(x.label()).tag == ':' for x in search_space]):
                    colon_index = [
                        Label(x.label()).tag == ':' for x in search_space
                    ].index(True)
                    search_space = search_space[:colon_index]
                not_a_modifiers = [
                    (i, x)
                    for i, x in enumerate(search_space, start=position[-1] + 1)
                    if not (is_verb_modifier(x) or Label(x.label()).tag in {
                        ',', 'CC', 'CONJP', 'PRN', '\'\'', '``', '.', '-LRB-',
                        '-RRB-'
                    } or Label(x.label()).tag.startswith('VB'))
                ]
                if any([
                        'PRD' in Label(x.label()).function
                        for i, x in not_a_modifiers
                ]):
                    # find the first PRD
                    prd_index = [
                        'PRD' in Label(x.label()).function
                        for i, x in not_a_modifiers
                    ].index(True)
                    not_a_modifier_index, not_a_modifier = not_a_modifiers[
                        prd_index]
                    not_a_modifier_label = Label(not_a_modifier.label())
                    subject_complement = {
                        'id':
                        get_tree_id(tree_id,
                                    parent_positons + [not_a_modifier_index]),
                        'text':
                        get_text_from_tree(not_a_modifier),
                        'plain_text':
                        get_plain_text_from_tree(not_a_modifier),
                        **get_subject_type(not_a_modifier),
                        'predicate_id':
                        get_tree_id(tree_id, predicate_absolute_positions),
                    }
                elif len(not_a_modifiers) == 1:
                    not_a_modifier_index, not_a_modifier = not_a_modifiers[0]
                    not_a_modifier_label = Label(not_a_modifier.label())
                    if (not_a_modifier_label.tag in {'S', 'SBAR', 'SQ'}
                            or (not_a_modifier_label.tag in {'NP', 'UCP'}
                                and len(not_a_modifier_label.function) == 0)):
                        object = {
                            'id':
                            get_tree_id(
                                tree_id,
                                parent_positons + [not_a_modifier_index]),
                            'text':
                            get_text_from_tree(not_a_modifier),
                            'plain_text':
                            get_plain_text_from_tree(not_a_modifier),
                            **get_subject_type(not_a_modifier),
                            'predicate_id':
                            get_tree_id(tree_id, predicate_absolute_positions),
                        }
                    else:
                        # outlier
                        # rich.print(not_a_modifiers)
                        # parent_tree.pretty_print(unicodelines=True)
                        # rich.print('outlier')
                        pass
                        # raise Exception('outlier')
                elif len(not_a_modifiers) == 2:
                    # IO + DO, DO + Complement
                    first_not_a_modifier_index, first_not_a_modifier = not_a_modifiers[
                        0]
                    second_not_a_modifier_index, second_not_a_modifier = not_a_modifiers[
                        1]
                    if Label(second_not_a_modifier.label()).tag == 'NP':
                        if Label(first_not_a_modifier.label()).tag == 'NP':
                            indirect_object = {
                                'id':
                                get_tree_id(
                                    tree_id, parent_positons +
                                    [first_not_a_modifier_index]),
                                'text':
                                get_text_from_tree(first_not_a_modifier),
                                'plain_text':
                                get_plain_text_from_tree(first_not_a_modifier),
                                **get_subject_type(first_not_a_modifier),
                                'predicate_id':
                                get_tree_id(tree_id,
                                            predicate_absolute_positions),
                            }
                            object = {
                                'id':
                                get_tree_id(
                                    tree_id, parent_positons +
                                    [second_not_a_modifier_index]),
                                'text':
                                get_text_from_tree(second_not_a_modifier),
                                'plain_text':
                                get_plain_text_from_tree(
                                    second_not_a_modifier),
                                **get_subject_type(second_not_a_modifier),
                                'predicate_id':
                                get_tree_id(tree_id,
                                            predicate_absolute_positions),
                            }
                        else:
                            # outlier
                            # rich.print(not_a_modifiers)
                            # parent_tree.pretty_print(unicodelines=True)
                            # rich.print('outlier: IO + DO')
                            pass
                            # raise Exception('outlier: IO + DO')
                    elif Label(second_not_a_modifier.label()).tag in {
                            'S', 'SBAR'
                    }:
                        object = {
                            'id':
                            get_tree_id(
                                tree_id, parent_positons +
                                [first_not_a_modifier_index]),
                            'text':
                            get_text_from_tree(first_not_a_modifier),
                            'plain_text':
                            get_plain_text_from_tree(first_not_a_modifier),
                            **get_subject_type(first_not_a_modifier),
                            'predicate_id':
                            get_tree_id(tree_id, predicate_absolute_positions),
                        }
                        complement = {
                            'id':
                            get_tree_id(
                                tree_id, parent_positons +
                                [second_not_a_modifier_index]),
                            'text':
                            get_text_from_tree(second_not_a_modifier),
                            'plain_text':
                            get_plain_text_from_tree(second_not_a_modifier),
                            **get_subject_type(second_not_a_modifier),
                            'predicate_id':
                            get_tree_id(tree_id, predicate_absolute_positions),
                        }
                    else:
                        # outlier
                        # parent_tree.pretty_print(unicodelines=True)
                        # rich.print(not_a_modifiers)
                        # rich.print('outlier: DO + Complement')
                        pass
                        # raise Exception('outlier: DO + Complement')
                if (object is not None or indirect_object is not None
                        or complement is not None
                        or subject_complement is not None):
                    result = {
                        'tree_id':
                        get_tree_id(tree_id, predicate_absolute_positions),
                        'extract_method':
                        self._extract_method_name,
                        'result': {}
                    }
                    if object is not None:
                        result['result']['object'] = object
                    if indirect_object is not None:
                        result['result']['indirect_object'] = indirect_object
                    if complement is not None:
                        result['result']['complement'] = complement
                    if subject_complement is not None:
                        result['result'][
                            'subject_complement'] = subject_complement
                    results.append(result)

        return {"results": results}
