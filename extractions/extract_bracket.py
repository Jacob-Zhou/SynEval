import nltk
import rich
from utils.extraction import TreeExtraction
from utils.fn import (EmptyType, Label, get_clause_type,
                      get_plain_text_from_tree, get_subject_type,
                      get_text_from_tree, search_bracket, get_tree_id)


class IdentifiedBracketExtraction(TreeExtraction):

    def extract_identified_bracket(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.identity_index is not None

        candidate_brackets = search_bracket(tree, label=label_fn)

        identity_brackets = {}
        for i, candidate_bracket in enumerate(candidate_brackets):
            tree_label = Label(candidate_bracket['tree'].label())
            # TODO: classify the identity bracket
            identity_brackets[tree_label.identity_index] = {
                'id': get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'text': get_text_from_tree(candidate_bracket['tree']),
                'plain_text':
                get_plain_text_from_tree(candidate_bracket['tree']),
            }

        # re-order the identity brackets
        identity_brackets = dict(
            sorted(identity_brackets.items(), key=lambda x: x[0]))

        return {
            "results": [{
                'tree_id': get_tree_id(tree_id, [0]),
                'extract_method': self._extract_method_name,
                'result': {
                    'identity_brackets': identity_brackets
                }
            }]
        }


class BarClauseExtraction(TreeExtraction):

    def extract_sbar_clause(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == 'SBAR'

        candidate_brackets = search_bracket(tree, label=label_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            candidate_bracket_label = Label(candidate_bracket['tree'].label())
            enclosed_clauses = search_bracket(
                list(candidate_bracket['tree']),
                label=lambda x: x.tag in {'S', 'SINV'},
                recursive=False,
            )
            if len(enclosed_clauses) == 0:
                bracket_pos_tags = candidate_bracket['tree'].pos()
                if len(bracket_pos_tags
                       ) == 1 and bracket_pos_tags[0][1] == '-NONE-':
                    leaf = EmptyType(bracket_pos_tags[0][0])
                    if leaf.reference_index is not None:
                        assert leaf.tag in {
                            '*ICH*', '*EXP*', '*RNR*', '*PPA*', '*T*'
                        }, rich.print(candidate_bracket)
                        results.append({
                            'tree_id':
                            get_tree_id(tree_id,
                                        candidate_bracket['positions'][1:]),
                            'extract_method':
                            self._extract_method_name,
                            'result': {
                                'type': 'subordinate_clause(empty)',
                                'enclosed_clause': {
                                    'id': None,
                                    'empty_type': leaf.tag,
                                    'trace': leaf.reference_index
                                }
                            }
                        })
            else:
                for enclosed_clause in enclosed_clauses:
                    enclosed_clause_label = Label(
                        enclosed_clause['tree'].label())
                    enclosed_clause_positions = candidate_bracket['positions'][
                        1:] + enclosed_clause['positions']
                    if enclosed_clause_label.tag == 'SINV' and 'ADV' not in candidate_bracket_label.function:
                        continue
                    subtype = get_clause_type(
                        candidate_bracket['tree'])['type']
                    results.append({
                        'tree_id':
                        get_tree_id(tree_id,
                                    candidate_bracket['positions'][1:]),
                        'extract_method':
                        self._extract_method_name,
                        'result': {
                            'type': 'subordinate_clause',
                            'subtype': subtype,
                            'enclosed_clause': {
                                'id':
                                get_tree_id(tree_id, enclosed_clause_positions)
                            }
                        }
                    })

        return {"results": results}

    def extract_sbarq_clause(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == 'SBARQ'

        candidate_brackets = search_bracket(tree, label=label_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            enclosed_clauses = search_bracket(
                list(candidate_bracket['tree']),
                label=lambda x: x.tag == {'SQ', 'S', 'FRAG'},
                recursive=False,
            )
            if len(enclosed_clauses) == 0:
                bracket_pos_tags = candidate_bracket['tree'].pos()
                if len(bracket_pos_tags
                       ) == 1 and bracket_pos_tags[0][1] == '-NONE-':
                    leaf = EmptyType(bracket_pos_tags[0][0])
                    if leaf.reference_index is not None:
                        assert leaf.tag in {'*T*', '*ICH*'
                                            }, rich.print(candidate_bracket)
                        results.append({
                            'tree_id':
                            get_tree_id(tree_id,
                                        candidate_bracket['positions'][1:]),
                            'extract_method':
                            self._extract_method_name,
                            'result': {
                                'type': 'direct_question(empty)',
                                'enclosed_clause': {
                                    'id': None,
                                    'empty_type': leaf.tag,
                                    'trace': leaf.reference_index
                                }
                            }
                        })
            else:
                # for enclosed_clause in enclosed_clauses:
                assert len(enclosed_clauses) == 1, rich.print(
                    candidate_bracket)
                enclosed_clause = enclosed_clauses[0]
                enclosed_clause_positions = candidate_bracket['positions'][
                    1:] + enclosed_clause['positions']
                subtype = get_clause_type(candidate_bracket['tree'])['type']
                results.append({
                    'tree_id':
                    get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                    'extract_method':
                    self._extract_method_name,
                    'result': {
                        'type': 'direct_question',
                        'subtype': subtype,
                        'enclosed_clause': {
                            'id': get_tree_id(tree_id,
                                              enclosed_clause_positions)
                        }
                    }
                })

        return {"results": results}


class PhraseExtraction(TreeExtraction):
    '''
        abstract class for phrase extraction
    '''
    target_tag = None
    target_type = None

    def __init__(self, method_name='run_extraction'):
        super().__init__(method_name)
        self.target_tag = self.__class__.target_tag
        self.target_type = self.__class__.target_type

    def extract_phrase(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == self.target_tag

        candidate_brackets = search_bracket(tree, label=label_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            children_labels = [
                Label(x.label()).tag for x in candidate_bracket['tree']
            ]
            type = self.target_type
            if 'CC' in children_labels or 'CONJP' in children_labels:
                type += "(compound)"
            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'type':
                    type,
                    'text':
                    get_text_from_tree(candidate_bracket['tree']),
                    'plain_text':
                    get_plain_text_from_tree(candidate_bracket['tree']),
                }
            })
        return {"results": results}


class ClauseExtraction(PhraseExtraction):
    target_tag = 'S'
    target_type = 'clause'

    def extract_phrase(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == self.target_tag

        candidate_brackets = search_bracket(tree, label=label_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            children_labels = [
                Label(x.label()).tag for x in candidate_bracket['tree']
            ]
            type = self.target_type
            if 'CC' in children_labels or 'CONJP' in children_labels:
                type += "(compound)"
            subtype = get_clause_type(candidate_bracket['tree'])['type']
            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'type':
                    type,
                    'subtype':
                    subtype,
                    'text':
                    get_text_from_tree(candidate_bracket['tree']),
                    'plain_text':
                    get_plain_text_from_tree(candidate_bracket['tree']),
                }
            })
        return {"results": results}

    def extract_coordination_clause(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == self.target_tag

        children_filter_fn = (lambda x: (
            (len(children_labels := {Label(x.label()).tag
                                     for x in x}) > 0) and
            (not children_labels.isdisjoint({'CC', 'CONJP'})) and
            (children_labels.isdisjoint({'NP', 'VP'})) and
            ('S' in children_labels)))

        candidate_brackets = search_bracket(
            tree, label=label_fn, children_filter_fn=children_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            candidate_tree = candidate_bracket['tree']
            candidate_tree_tags = candidate_tree.pos()
            rnr_indices = []
            for word, tag in candidate_tree_tags:
                if (tag == '-NONE-'
                        and (word := EmptyType(word)).tag == '*RNR*'):
                    if word.reference_index is not None:
                        rnr_indices.append(word.reference_index)
            if any([
                    Label(child.label()).identity_index in rnr_indices
                    for child in candidate_tree
            ]):
                continue

            coordinated_noun_phrases = []
            for child_index, child in enumerate(candidate_tree):
                if (child_label := Label(child.label())).tag in {
                        'S', 'SINV', 'SQ'
                } and len(child_label.function) == 0:
                    # can not have any function tags
                    coordinated_noun_phrases.append({
                        'id':
                        get_tree_id(
                            tree_id, candidate_bracket['positions'][1:] +
                            [child_index]),
                        'text':
                        get_text_from_tree(child),
                        'plain_text':
                        get_plain_text_from_tree(child),
                    })

            if len(coordinated_noun_phrases) < 2:
                continue

            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'coordinated_clauses': coordinated_noun_phrases
                }
            })

        return {"results": results}


class NounPhraseExtraction(TreeExtraction):
    target_tag = 'NP'
    target_type = 'noun_phrase'

    def extract_phrase(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == self.target_tag

        candidate_brackets = search_bracket(tree, label=label_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            children_labels = [
                Label(x.label()).tag for x in candidate_bracket['tree']
            ]
            type = self.target_type
            if ',' in children_labels and ('CC' not in children_labels
                                           and 'CONJP' not in children_labels):
                type += '(non-restrictive)'
            elif ('CC' in children_labels or 'CONJP' in children_labels):
                # donot extract NP with appositive modifier or non-restrictive modifier
                type += '(compound)'
            subtype = get_subject_type(candidate_bracket['tree'])['type']
            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'type':
                    type,
                    'subtype':
                    subtype,
                    'text':
                    get_text_from_tree(candidate_bracket['tree']),
                    'plain_text':
                    get_plain_text_from_tree(candidate_bracket['tree']),
                }
            })
        return {"results": results}

    def extract_coordination_noun_phrase(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == self.target_tag

        parent_filter_fn = (
            lambda x: Label(x[-1].label()).tag not in {'NP', 'UCP'})

        children_filter_fn = (lambda x: (
            ({'NP', 'DT', 'CC', 'CONJP', ',', ':', '``', "''"}.issuperset(
                children_labels := {Label(x.label()).tag
                                    for x in x})) and
            (not children_labels.isdisjoint({'CC', 'CONJP'})) and
            ('NP' in children_labels)))

        candidate_brackets = search_bracket(
            tree,
            label=label_fn,
            parent_filter_fn=parent_filter_fn,
            children_filter_fn=children_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            candidate_tree = candidate_bracket['tree']
            children_labels = {
                Label(child.label()).tag
                for child in candidate_tree
            }
            if (('DT' in children_labels)
                    and Label(candidate_tree[0].label()).tag != 'DT'):
                continue

            coordinated_noun_phrases = []
            for child_index, child in enumerate(candidate_tree):
                if Label(child.label()).tag == 'NP':
                    coordinated_noun_phrases.append({
                        'id':
                        get_tree_id(
                            tree_id, candidate_bracket['positions'][1:] +
                            [child_index]),
                        'text':
                        get_text_from_tree(child),
                        'plain_text':
                        get_plain_text_from_tree(child),
                    })

            if len(coordinated_noun_phrases) < 2:
                continue

            results.append({
                'tree_id':
                get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                'extract_method':
                self._extract_method_name,
                'result': {
                    'coordinated_noun_phrases': coordinated_noun_phrases
                }
            })

        return {"results": results}


class PrepositionalPhraseExtraction(PhraseExtraction):
    target_tag = 'PP'
    target_type = 'prepositional_phrase'


class AdjectivePhraseExtraction(PhraseExtraction):
    target_tag = 'ADJP'
    target_type = 'adjective_phrase'


class AdverbPhraseExtraction(PhraseExtraction):
    target_tag = 'ADVP'
    target_type = 'adverb_phrase'


class QuantifierPhraseExtraction(PhraseExtraction):
    target_tag = 'QP'
    target_type = 'quantifier_phrase'