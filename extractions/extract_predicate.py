from collections import Counter
import nltk
import rich
from utils.extraction import TreeExtraction
from utils.fn import (EmptyType, Label, get_plain_text_from_tree,
                      get_text_from_tree, get_text_from_verb_sequence,
                      get_verb_type_from_verb_sequence, flatten_verb_phrase,
                      get_subject_type, is_declarative_sentence,
                      is_it_cleft_sentence, is_passive_sentence,
                      normalize_word, search_bracket, clean_pos, get_tree_id)

# 2.1 Predicate Extraction
# 2.1.1 Verb Phrase Predicate, VP can be a direct child of S, NP(non-finite, post-modified), SINV and SQ
# 2.1.2 Predicate that is not a verb phrase


class PredicateExtraction(TreeExtraction):

    def extract_verb_phrase_predicate_in_declarative_clause(
            self, tree_id: str, tree: nltk.Tree):
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
            clause_positons = candidate_bracket['positions'][1:-1]
            verb_phrase_results = []
            predicate_results = []
            for verb_sequence in verb_sequences:
                if len(verb_sequence) == 0:
                    # gapping VP
                    continue
                verb_type = get_verb_type_from_verb_sequence(verb_sequence)
                subtype = verb_type.pop('type')
                verb_type['type'] = 'predicate'
                verb_word, position, *_ = verb_sequence[-1]
                position = candidate_bracket['positions'][1:] + position
                if verb_word is not None:
                    predicate_result = {
                        'id':
                        get_tree_id(tree_id, position),
                        'text':
                        get_text_from_verb_sequence(verb_sequence),
                        'plain_text':
                        get_text_from_verb_sequence(verb_sequence,
                                                    keep_adverb=True),
                        **verb_type,
                        'subtype':
                        subtype,
                        # 'positions': position
                    }
                else:
                    predicate_result = {
                        'id': get_tree_id(tree_id, position),
                        'text': None,  # TODO rebuild by sibilings
                        'plain_text': None,
                        **verb_type,
                        'subtype': subtype,
                        # 'positions': position
                    }
                predicate_results.append(predicate_result)
                verb_phrase_results.append(predicate_result)
                results.append({
                    'tree_id': get_tree_id(tree_id, position),
                    'extract_method': self._extract_method_name,
                    'result': predicate_result
                })
            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'text':
                    get_text_from_tree(candidate_bracket['tree']),
                    'plain_text':
                    get_plain_text_from_tree(candidate_bracket['tree']),
                    'type':
                    'verb_phrase',
                    'verb_sequences':
                    verb_phrase_results
                }
            })
            results.append({
                'tree_id': get_tree_id(tree_id, clause_positons),
                'extract_method': self._extract_method_name,
                'result': {
                    'predicates': predicate_results
                }
            })

        return {"results": results}
