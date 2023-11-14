import copy
import nltk
import rich
from utils.extraction import TreeExtraction
from utils.fn import (Label, get_clause_type, get_noun_modifier_type,
                      get_noun_phrase_type, get_plain_text_from_tree,
                      get_text_from_tree, get_verb_modifier_type,
                      flatten_verb_phrase, has_elder_noun_sibling,
                      is_verb_modifier, search_bracket, get_tree_id)
from utils.common import relative_pronouns

# 3.1 Modifier Extraction
# 3.1.1 noun modifier
# 3.1.2 verb modifier


class NounModifierExtraction(TreeExtraction):

    def extract_noun_postmodifier(self, tree_id: str, tree: nltk.Tree):
        # NP that is not possessive
        label_fn = lambda x: x.tag == 'NP'
        self_filter_fn = (
            lambda x: get_noun_phrase_type(x)['type'] != 'possessive')

        # a child of NP
        parent_filter_fn = (lambda x: Label(x[-1].label()).tag == 'NP')

        younger_siblings_filter_fn = (lambda x: len(x) > 0 and not {
            Label(child.label()).tag
            for child in x
        }.isdisjoint(
            {'PP', 'ADJP', 'ADVP', 'NP', 'SBAR', 'VP', 'S', 'RRC', ','}))

        elder_siblings_filter_fn = (lambda x: not has_elder_noun_sibling(x))

        candidate_brackets = search_bracket(
            tree,
            label=label_fn,
            self_filter_fn=self_filter_fn,
            parent_filter_fn=parent_filter_fn,
            younger_siblings_filter_fn=younger_siblings_filter_fn,
            elder_siblings_filter_fn=elder_siblings_filter_fn)

        # NP that is possessive
        self_filter_fn = (
            lambda x: get_noun_phrase_type(x)['type'] == 'possessive')

        younger_siblings_filter_fn = (
            lambda x: len(x) > 0 and (not any(
                Label(child.label()).tag in {'NP', 'NN', 'NNS', 'NNP', 'NNPS'}
                for child in x)) and
            (not {Label(child.label()).tag
                  for child in x}.isdisjoint(
                      {'PP', 'ADJP', 'ADVP', 'SBAR', 'VP', 'S', 'RRC', ','})))

        # avoid mistaking put postmodifiers to appositions
        elder_siblings_filter_fn = (
            lambda x: {Label(child.label()).tag
                       for child in x}.isdisjoint({'NP'}))

        candidate_brackets += search_bracket(
            tree,
            label=label_fn,
            self_filter_fn=self_filter_fn,
            parent_filter_fn=parent_filter_fn,
            younger_siblings_filter_fn=younger_siblings_filter_fn,
            elder_siblings_filter_fn=elder_siblings_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            parent_phrase = tree[candidate_bracket['positions'][1:-1]]
            younger_siblings = parent_phrase[
                candidate_bracket['positions'][-1] + 1:]
            if not {Label(child.label()).tag
                    for child in younger_siblings}.isdisjoint({'CC', 'CONJP'}):
                # skip coordination NP
                continue
            modifiers = []
            appositions = []
            for child_index, child in enumerate(younger_siblings):
                child_position = copy.deepcopy(
                    candidate_bracket['positions'][1:])
                child_position[-1] += child_index + 1
                if Label(child.label()).tag in {
                        'PP', 'ADJP', 'ADVP', 'SBAR', 'VP', 'S', 'RRC'
                }:
                    child_label = Label(child.label())
                    modifier = {
                        'id': get_tree_id(tree_id, child_position),
                        'text': get_text_from_tree(child),
                        'plain_text': get_plain_text_from_tree(child),
                        **get_noun_modifier_type(child),
                        # 'positions': child_position
                    }
                    if younger_siblings[child_index - 1].pos()[0] == (',',
                                                                      ','):
                        if child_label.tag == 'SBAR':
                            modifier['type'] += '(non-restrictive)'
                            modifiers.append(modifier)
                        else:
                            appositions.append(modifier)
                    else:
                        modifiers.append(modifier)
                    # add complementizer mapping
                    if child_label.tag == 'SBAR':
                        refer_noun = candidate_bracket['tree']
                        refer_noun_id = get_tree_id(
                            tree_id, candidate_bracket['positions'][1:])
                        if 'SBAR' in [Label(x.label()).tag for x in child]:
                            sbars = [(i, x) for i, x in enumerate(child)
                                     if Label(x.label()).tag == 'SBAR']
                        else:
                            sbars = [(None, child)]
                        # SBAR -> (WH* S)
                        for sbar_index, sbar in sbars:
                            wh_element = sbar[0]
                            if wh_element.label() == '-NONE-':
                                continue
                            wh_element_label = Label(wh_element.label())
                            if wh_element_label.identity_index is None:
                                continue
                            assert wh_element.label().startswith(
                                'WH'), sbar.pretty_print(unicodelines=True)
                            specified_wh_element = copy.deepcopy(wh_element)
                            for b in list(specified_wh_element.subtrees()):
                                if isinstance(b, nltk.Tree) and not isinstance(
                                        b[0], str):
                                    for i, c in enumerate(b):
                                        if (len(c.leaves()) == 1
                                                and c.leaves()[0].lower()
                                                in relative_pronouns | {'0'}):
                                            b[i] = refer_noun
                                            break
                            if sbar_index is None:
                                sbar_position = child_position
                            else:
                                sbar_position = child_position + [sbar_index]
                            results.append({
                                "tree_id":
                                get_tree_id(tree_id, sbar_position + [0]),
                                "extract_method":
                                self._extract_method_name,
                                "result": {
                                    "type":
                                    "complementizer",
                                    "text":
                                    get_text_from_tree(wh_element),
                                    "plain_text":
                                    get_plain_text_from_tree(wh_element),
                                    "specified_plain_text":
                                    get_plain_text_from_tree(
                                        specified_wh_element),
                                    "identity_index":
                                    Label(wh_element.label()).identity_index,
                                    "refer_noun_id":
                                    refer_noun_id
                                }
                            })
                elif Label(child.label()).tag == 'NP':
                    if child_index == 0:
                        continue
                    if younger_siblings[child_index - 1].pos()[0] == (',',
                                                                      ','):
                        # after a comma: apposition
                        appositions.append({
                            'id':
                            get_tree_id(tree_id, child_position),
                            'text':
                            get_text_from_tree(child),
                            'plain_text':
                            get_plain_text_from_tree(child),
                            **get_noun_modifier_type(child),
                        })
                    elif (len(child) == 1 and child.pos()[0][1] == 'PRP'
                          and 'sel' in child.leaves()[0].lower()):
                        modifiers.append({
                            'id':
                            get_tree_id(tree_id, child_position),
                            'text':
                            get_text_from_tree(child),
                            'plain_text':
                            get_plain_text_from_tree(child),
                            'type':
                            'reflexive_pronoun',
                        })
                    elif (len(child) == 1
                          and child.leaves()[0].lower() == 'all'):
                        modifiers.append({
                            'id':
                            get_tree_id(tree_id, child_position),
                            'text':
                            get_text_from_tree(child),
                            'plain_text':
                            get_plain_text_from_tree(child),
                            'type':
                            'quantifier',
                        })
            result = {}
            if len(modifiers) > 0:
                result['modifiers'] = modifiers
            if len(appositions) > 0:
                result['appositions'] = appositions
            if len(result) > 0:
                results.extend([{
                    'tree_id':
                    get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                    'extract_method':
                    self._extract_method_name,
                    'result':
                    result
                }, {
                    'tree_id':
                    get_tree_id(tree_id, candidate_bracket['positions'][1:-1]),
                    'extract_method':
                    self._extract_method_name,
                    'result': {
                        'main_noun': {
                            'id':
                            get_tree_id(tree_id,
                                        candidate_bracket['positions'][1:]),
                            'text':
                            get_text_from_tree(candidate_bracket['tree']),
                            'plain_text':
                            get_plain_text_from_tree(
                                candidate_bracket['tree']),
                        }
                    }
                }])
        return {"results": results}

    def extract_noun_complement(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == 'NP'
        # Only clausal complement of NP are placed inside NP

        # allow NP to have a possessive NP child

        children_filter_fn = (lambda x: (
            (not {Label(child.label()).tag
                  for child in x}.isdisjoint({'SBAR', 'S'})) and (sum([
                      get_noun_phrase_type(child)['type'] != 'possessive'
                      for child in x if Label(child.label()).tag == 'NP'
                  ]) == 0)))

        candidate_brackets = search_bracket(
            tree, label=label_fn, children_filter_fn=children_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            # The candidate bracket is the highest NP
            noun_phrase = candidate_bracket['tree']
            complements = []
            not_a_complement = False
            for child_index, child in enumerate(noun_phrase):
                if Label(child.label()).tag in {'SBAR', 'S'}:
                    if child_index == 0:
                        not_a_complement = True
                        break
                    child_position = candidate_bracket['positions'] + [
                        child_index
                    ]
                    complements.append({
                        'id':
                        get_tree_id(tree_id, child_position),
                        'text':
                        get_text_from_tree(child),
                        'plain_text':
                        get_plain_text_from_tree(child),
                        **get_clause_type(child),
                        # 'positions': child_position
                    })
            if not not_a_complement and len(complements) > 0:
                results.append({
                    'tree_id':
                    get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                    'extract_method':
                    self._extract_method_name,
                    'result': {
                        'complements': complements
                    }
                })

        return {"results": results}

    def _extract_possessive(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag in {'NP', 'NN'}

        elder_siblings_filter_fn = (lambda x: any([
            get_noun_phrase_type(child)['type'] == 'possessive' for child in x
            if Label(child.label()).tag == 'NP'
        ]))

        candidate_brackets = search_bracket(
            tree,
            label=label_fn,
            elder_siblings_filter_fn=elder_siblings_filter_fn)

        rich.print(candidate_brackets)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            # The candidate bracket is the highest NP
            possessives = []
            parent_phrase = tree[candidate_bracket['positions'][1:-1]]
            elder_siblings = parent_phrase[:candidate_bracket['positions'][-1]]
            for elder_sibling_index, elder_sibling in reversed(
                    list(enumerate(elder_siblings))):
                elder_sibling_position = candidate_bracket[
                    'positions'][:-1] + [elder_sibling_index]
                if Label(elder_sibling.label()
                         ).tag == 'NP' and get_noun_phrase_type(
                             elder_sibling)['type'] == 'possessive':
                    possessives.append({
                        'id':
                        get_tree_id(tree_id, elder_sibling_position),
                        'text':
                        get_text_from_tree(elder_sibling),
                        'plain_text':
                        get_plain_text_from_tree(elder_sibling),
                        **get_noun_phrase_type(elder_sibling),
                        # 'positions': elder_sibling_position
                    })
                else:
                    break
            if len(possessives) > 0:
                results.append({
                    'tree_id':
                    get_tree_id(tree_id, candidate_bracket['positions'][1:]),
                    'extract_method':
                    self._extract_method_name,
                    'result': {
                        'possessives': possessives
                    }
                })

        return {"results": results}


# 3.1.2 verb modifier
# 3.1.2.1 adverbial modifier
# 3.1.2.2 nominal modifier
# 3.1.2.3 prepositional modifier
# 3.1.2.4 non-finite modifier (S-ADV)
# 3.1.2.5 subordinate clause modifier (SBAR)


class VerbModifierExtraction(TreeExtraction):

    def extract_verb_modifier(self, tree_id: str, tree: nltk.Tree):
        label_fn = lambda x: x.tag == 'VP'

        # highest VP
        parent_filter_fn = (lambda x: Label(x[-1].label()).tag not in {'VP'})

        # must have a VP child
        candidate_brackets = search_bracket(tree,
                                            label=label_fn,
                                            parent_filter_fn=parent_filter_fn)

        results = []
        for i, candidate_bracket in enumerate(candidate_brackets):
            # The candidate bracket is the highest VP
            pre_modifiers = []
            verb_phrase = candidate_bracket['tree']
            candidate_bracket_positions = candidate_bracket['positions'][1:]
            parent_phrase = tree[candidate_bracket_positions[:-1]]
            after_subject = False
            for node_index, node in enumerate(
                    parent_phrase[:candidate_bracket_positions[-1]]):
                node_position = candidate_bracket_positions[:-1] + [node_index]
                # the adverbial modifier before the subject is modifing the whole sentence
                if 'SBJ' in Label(node.label()).function:
                    after_subject = True
                    continue
                if after_subject and is_verb_modifier(node):
                    pre_modifiers.append({
                        'id':
                        get_tree_id(tree_id, node_position),
                        'text':
                        get_text_from_tree(node),
                        'plain_text':
                        get_plain_text_from_tree(node),
                        **get_verb_modifier_type(node),
                        # 'positions': node_position
                    })

            flattened_verb_phrase = flatten_verb_phrase(verb_phrase)
            for verb_sequence in flattened_verb_phrase:
                if len(verb_sequence) == 0:
                    # gapping VP
                    continue
                main_verb_word, main_verb_position, *_ = verb_sequence[-1]
                main_verb_position = candidate_bracket['positions'][
                    1:] + main_verb_position
                # search for adverbial modifiers
                modifiers = copy.deepcopy(pre_modifiers)
                appositions = []
                assert len(verb_sequence) == len(verb_sequence[-1][1])
                for ((verb_word, verb_position, siblings),
                     verb_phrase_position) in zip(verb_sequence,
                                                  verb_sequence[-1][1]):
                    parent_position = verb_position[:-1]
                    verb_position = verb_position[-1]
                    verb_phrases_in_siblings = [
                        sibling_position
                        for sibling_position, sibling in enumerate(siblings)
                        if Label(sibling.label()).tag == 'VP'
                    ]
                    search_space = []
                    if verb_position is None:
                        if len(verb_phrases_in_siblings) == 1:
                            # [ADVP/RB] VP   VP [ADVP]
                            # ^all           ^all
                            search_space += list(enumerate(siblings))
                        elif len(verb_phrases_in_siblings) > 1:
                            # [ADVP/RB] VP   [ADVP/RB] VP [ADVP]
                            # ^all if ADVP   ^to right    ^all
                            # ^right if RB
                            search_space += [(i, x) for i, x in enumerate(
                                siblings[:verb_phrases_in_siblings[0]])
                                             if Label(x.label()).tag == 'ADVP']
                            search_space += list(
                                enumerate(
                                    siblings[verb_phrases_in_siblings[-1] +
                                             1:],
                                    start=verb_phrases_in_siblings[-1] + 1))
                            verb_phrase_index = verb_phrases_in_siblings.index(
                                verb_phrase_position)
                            if verb_phrase_index > 0:
                                search_space += list(
                                    enumerate(
                                        siblings[verb_phrases_in_siblings[
                                            verb_phrase_index - 1] +
                                                 1:verb_phrase_position],
                                        start=verb_phrases_in_siblings[
                                            verb_phrase_index - 1] + 1))
                            elif verb_phrase_index == 0:
                                search_space += [
                                    (i, x) for i, x in enumerate(
                                        siblings[:verb_phrases_in_siblings[0]])
                                    if Label(x.label()).tag == 'RB'
                                ]
                        else:
                            continue
                    else:
                        if len(verb_phrases_in_siblings) <= 1:
                            # 0:
                            #    [ADVP/RB] verb [ADVP/RB]
                            #    ^all           ^all
                            # 1:
                            #    [ADVP/RB] verb [ADVP/RB] VP [ADVP]
                            #    ^all           ^all      ^all
                            search_space += list(enumerate(siblings))
                        else:
                            # [ADVP/RB] verb [ADVP/RB] VP [ADVP/RB] VP [ADVP]
                            # ^all           ^all         ^to right    ^all
                            search_space += list(
                                enumerate(
                                    siblings[:verb_phrases_in_siblings[0]]))
                            search_space += list(
                                enumerate(
                                    siblings[verb_phrases_in_siblings[-1] +
                                             1:],
                                    start=verb_phrases_in_siblings[-1] + 1))
                            verb_phrase_index = verb_phrases_in_siblings.index(
                                verb_phrase_position)
                            if verb_phrase_index > 0:
                                search_space += list(
                                    enumerate(
                                        siblings[verb_phrases_in_siblings[
                                            verb_phrase_index - 1] +
                                                 1:verb_phrase_position],
                                        start=verb_phrases_in_siblings[
                                            verb_phrase_index - 1] + 1))
                    # search for adverbial modifiers
                    for node_index, node in search_space:
                        node_position = (candidate_bracket['positions'][1:] +
                                         parent_position + [node_index])
                        node_label = Label(node.label())
                        is_modifier = is_verb_modifier(node)
                        if verb_word is None and node_label.tag == 'NP':
                            # handle apposition NP
                            appositions.append({
                                'id':
                                get_tree_id(tree_id, node_position),
                                'text':
                                get_text_from_tree(node),
                                'plain_text':
                                get_plain_text_from_tree(node),
                                **get_verb_modifier_type(node),
                                # 'positions': node_position
                            })
                        # another case is: SBAR,
                        if is_modifier:
                            modifiers.append({
                                'id':
                                get_tree_id(tree_id, node_position),
                                'text':
                                get_text_from_tree(node),
                                'plain_text':
                                get_plain_text_from_tree(node),
                                **get_verb_modifier_type(node),
                                # 'positions': node_position
                            })
                result = {}
                if len(modifiers) > 0:
                    result['modifiers'] = modifiers
                if len(appositions) > 0:
                    result['appositions'] = appositions
                if len(result) > 0:
                    results.append({
                        'tree_id':
                        get_tree_id(tree_id, main_verb_position),
                        'extract_method':
                        self._extract_method_name,
                        'result':
                        result
                    })

        return {"results": results}
