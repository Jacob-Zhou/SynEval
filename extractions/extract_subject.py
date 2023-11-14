from collections import Counter
import copy
import nltk
import rich
from utils.extraction import TreeExtraction
from utils.fn import (Label, get_plain_text_from_tree, get_text_from_tree,
                      get_verb_type_from_verb_sequence, flatten_verb_phrase,
                      get_subject_type, is_declarative_sentence,
                      is_it_cleft_sentence, is_passive_sentence,
                      search_bracket, clean_pos, get_tree_id)

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


class SubjectExtraction(TreeExtraction):

    def extract_surface_subject_in_declarative_clause(self, tree_id: str,
                                                      tree: nltk.Tree):
        '''
            the extraction method for 1.1.1, 1.1.2 and 1.1.4 subject in declarative sentence
            :param tree: a nltk.Tree object
        '''
        # filter functions
        label_fn = lambda x: 'SBJ' in x.function

        parent_filter_fn = lambda x: Label(x[-1].label()).tag == 'S'

        younger_siblings_filter_fn = (
            lambda x: 'VP' in {Label(child.label()).tag
                               for child in x})  # must have a VP sibling

        candidate_brackets = search_bracket(
            tree,
            label=label_fn,
            parent_filter_fn=parent_filter_fn,
            younger_siblings_filter_fn=younger_siblings_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            parent_positons = candidate_bracket['positions'][1:-1]
            subject_type = get_subject_type(candidate_bracket['tree'])
            result = {
                'tree_id': get_tree_id(tree_id, parent_positons),
                'extract_method': self._extract_method_name,
                'result': {
                    'surface_subject': {
                        'id':
                        get_tree_id(tree_id,
                                    candidate_bracket['positions'][1:]),
                        'text':
                        get_text_from_tree(candidate_bracket['tree']),
                        'plain_text':
                        get_plain_text_from_tree(candidate_bracket['tree']),
                        **subject_type,
                        # 'positions': candidate_bracket['positions'][1:]
                    }
                }
            }
            if 'CLF' in Label(tree[parent_positons].label()).function:
                result['result']['surface_subject']['type'] = 'it-cleft'
            results.append(result)
        return {"results": results}

    def extract_logical_subject_in_declarative_clause(self, tree_id: str,
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
            logical_subject_results = []
            for verb_sequence in verb_sequences:
                verb_type = get_verb_type_from_verb_sequence(verb_sequence)
                if verb_type['voice'] == 'passive':
                    verb_word, position, siblings = verb_sequence[-1]
                    younger_siblings = siblings[position[-1] + 1:]
                    position = candidate_bracket['positions'][1:] + position
                    logical_subject = None
                    # for youngest_sibling in youngest_siblings:
                    for i, younger_sibling in enumerate(younger_siblings):
                        youngest_sibling_label = Label(younger_sibling.label())
                        if (youngest_sibling_label.tag == 'PP' and
                                younger_sibling.leaves()[0].lower() == 'by'):
                            if 'LGS' in youngest_sibling_label.function:
                                logical_subject_position = [
                                    Label(x.label()).tag == 'NP'
                                    for x in younger_sibling
                                ]
                            else:
                                logical_subject_position = [
                                    'LGS' in Label(x.label()).function
                                    for x in younger_sibling
                                ]
                            if any(logical_subject_position):
                                logical_subject_position = logical_subject_position.index(
                                    True)
                                logical_subject = younger_sibling[
                                    logical_subject_position]
                                break
                    if logical_subject is not None:
                        predicate_absolute_positions = position
                        logical_subject_absolute_positions = copy.deepcopy(
                            position)
                        logical_subject_absolute_positions[-1] += i + 1
                        logical_subject_absolute_positions.append(
                            logical_subject_position)
                        logical_subject_results.append({
                            'id':
                            get_tree_id(tree_id,
                                        logical_subject_absolute_positions),
                            'text':
                            get_text_from_tree(logical_subject),
                            'plain_text':
                            get_plain_text_from_tree(logical_subject),
                            **get_subject_type(logical_subject),
                            # 'positions': logical_subject_absolute_positions,
                            'predicate_id':
                            get_tree_id(tree_id, predicate_absolute_positions),
                        })
                if len(logical_subject_results) > 0:
                    results.append({
                        'tree_id':
                        get_tree_id(tree_id, predicate_absolute_positions),
                        'extract_method':
                        self._extract_method_name,
                        'result': {
                            'logical_subjects': logical_subject_results
                        }
                    })

        return {"results": results}
