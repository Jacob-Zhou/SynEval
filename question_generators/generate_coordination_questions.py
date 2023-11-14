from collections import Counter
import copy
import random
import nltk
from utils.extraction import QuestionGeneration
from utils.common import random_pronouns, be_verbs, random_verbs


class CoordinationQuestionGeneration(QuestionGeneration):

    def pre_process(self, *args, **kwds):
        args, kwds = super().pre_process(*args, **kwds)
        json_obj = kwds['json_obj']
        self.noun_pharses = [(node_id, node_info)
                             for node_id, node_info in json_obj.items()
                             if 'noun_phrase' in node_info['type']]
        self.noun_phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if 'noun_phrase' in node_info['type']
            and node_info['plain_text'] is not None
        ])
        self.verb_phrases = [(node_id, node_info)
                             for node_id, node_info in json_obj.items()
                             if (node_info['type'].startswith('predicate')
                                 and node_info['plain_text'] is not None)]
        self.verb_phrase_text_counter = Counter([
            node_info['text'].lower().strip()
            for _, node_info in json_obj.items()
            if node_info['type'].startswith('predicate')
            and node_info['text'] is not None
        ])
        return args, kwds

    def build_options(self, correct_answer, remove_phrases, coordinated_type,
                      **kwds):
        options = [correct_answer]
        if coordinated_type == 'noun_phrase':
            random_phrases = copy.deepcopy(self.noun_pharses)
        else:
            random_phrases = copy.deepcopy(self.verb_phrases)
        random.shuffle(random_phrases)
        # remove the subject itself, its ancestors and its descendants
        for _, random_phrase in random_phrases:
            random_text = random_phrase['plain_text'].lower().strip()
            if random_text in remove_phrases:
                continue
            if random_text != '' and random_text not in options:
                options.append(random_text)
                if len(options) >= 4:
                    break

        if coordinated_type == 'noun_phrase':
            shuffled_random_pronouns = copy.deepcopy(random_pronouns)
            random.shuffle(shuffled_random_pronouns)
            for pronoun in shuffled_random_pronouns:
                if len(options) >= 4:
                    break
                if pronoun not in options:
                    options.append(pronoun)
        else:
            shuffled_random_verbs = copy.deepcopy(random_verbs)
            random.shuffle(shuffled_random_verbs)
            for verb in shuffled_random_verbs:
                if len(options) >= 4:
                    break
                if verb not in options:
                    options.append(verb)

        return options

    def generate_coordinated_noun_phrase_question(self, tree: nltk.Tree,
                                                  json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, which of the following is coordinated with the noun phrase “{noun_phrase}”?",
        ]
        self.question_templates['yes_no'] = [
            "In the above sentence, the noun phrase “{noun_phrase}” is coordinated with noun phrase “{correct_answer}”.",
            "<NEG> In the above sentence, the noun phrase “{noun_phrase}” is not coordinated with noun phrase “{correct_answer}”.",
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            if 'coordinated_noun_phrases' not in node_info:
                continue
            coordinated_noun_phrases = node_info['coordinated_noun_phrases']
            coordinated_noun_phrase_texts = [
                coordinated_noun_phrase['plain_text']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is empty, skip
            if (None in coordinated_noun_phrase_texts
                    or '' in coordinated_noun_phrase_texts):
                continue
            coordinated_noun_phrase_ids = [
                coordinated_noun_phrase['id']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is not in the json_obj, skip
            if any([
                    coordinated_noun_phrase_id not in json_obj for
                    coordinated_noun_phrase_id in coordinated_noun_phrase_ids
            ]):
                continue
            if len(set(coordinated_noun_phrase_texts)) < len(
                    coordinated_noun_phrase_texts):
                # the coordinated noun phrases are ambiguous
                continue
            if len(coordinated_noun_phrases) < 2:
                # there is only one noun phrase, skip
                continue
            for i, coordinated_noun_phrase_i in enumerate(
                    coordinated_noun_phrases):
                for j, coordinated_noun_phrase_j in enumerate(
                        coordinated_noun_phrases):
                    if i == j:
                        continue
                    noun_phrase_i_text = coordinated_noun_phrase_i[
                        'plain_text'].strip().lower()
                    if self.noun_phrase_text_counter[noun_phrase_i_text] > 1:
                        # the noun phrase is ambiguous
                        continue
                    noun_phrase_j_text = coordinated_noun_phrase_j[
                        'plain_text'].strip().lower()
                    noun_phrase_i_id = coordinated_noun_phrase_i['id']
                    noun_phrase_j_id = coordinated_noun_phrase_j['id']
                    current_question_tags = ["coordinated_noun_phrase"]
                    with_modifier = {'i': 'no', 'j': 'no'}
                    for id, prefix in [(noun_phrase_i_id, 'i'),
                                       (noun_phrase_j_id, 'j')]:
                        if ('main_noun' in json_obj[id]
                                and json_obj[id]['main_noun'] is not None and
                                json_obj[id]['main_noun']['id'] in json_obj):
                            main_noun_id = json_obj[id]['main_noun']['id']
                            if 'modifiers' not in json_obj[main_noun_id]:
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:no"
                                )
                                continue
                            else:
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:yes"
                                )
                                with_modifier[prefix] = 'yes'
                            if any([
                                    modifier['type'] == 'prepositional_phrase'
                                    for modifier in json_obj[main_noun_id]
                                ['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                                )
                            for clause in [
                                    'relative_clause',
                                    'reduced_relative_clause', 'wh_clause',
                                    'that_clause',
                                    'coordinated_subordinate_clause'
                            ]:
                                if any([
                                        clause in modifier['type'] for modifier
                                        in json_obj[main_noun_id]['modifiers']
                                ]):
                                    current_question_tags.append(
                                        f"coordinated_noun_phrase:with_modifier_{prefix}:with_{clause}"
                                    )
                    current_question_tags.append(
                        f"coordinated_noun_phrase:with_modifier:{with_modifier['i']}:{with_modifier['j']}"
                    )

                    instances.extend(
                        self.build_questions(
                            ingredient={
                                'noun_phrase': noun_phrase_i_text,
                            },
                            correct_answer=noun_phrase_j_text,
                            question_tags=current_question_tags,
                            question_source=(node_id, noun_phrase_i_id),
                            node_id=node_id,
                            remove_phrases=coordinated_noun_phrase_texts,
                            coordinated_type='noun_phrase',
                            question_infos={
                                "knowledge_point": "coordinated_phrase"
                            }))

        return {"instances": instances}

    def generate_coordinated_noun_phrase_question_incorrect_answer_yn(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['yes_no'] = [
            "In the above sentence, the noun phrase “{noun_phrase}” is coordinated with noun phrase “{incorrect_answer}”.",
            "<NEG> In the above sentence, the noun phrase “{noun_phrase}” is not coordinated with noun phrase “{incorrect_answer}”.",
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            if 'coordinated_noun_phrases' not in node_info:
                continue
            coordinated_noun_phrases = node_info['coordinated_noun_phrases']
            coordinated_noun_phrase_texts = [
                coordinated_noun_phrase['plain_text']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is empty, skip
            if (None in coordinated_noun_phrase_texts
                    or '' in coordinated_noun_phrase_texts):
                continue
            coordinated_noun_phrase_ids = [
                coordinated_noun_phrase['id']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is not in the json_obj, skip
            if any([
                    coordinated_noun_phrase_id not in json_obj for
                    coordinated_noun_phrase_id in coordinated_noun_phrase_ids
            ]):
                continue
            for i, coordinated_noun_phrase_i in enumerate(
                    coordinated_noun_phrases):
                noun_phrase_i_text = coordinated_noun_phrase_i[
                    'plain_text'].strip().lower()
                if self.noun_phrase_text_counter[noun_phrase_i_text] > 1:
                    # the noun phrase is ambiguous
                    continue
                noun_phrase_i_id = coordinated_noun_phrase_i['id']
                current_question_tags = ["coordinated_noun_phrase"]
                for id, prefix in [(noun_phrase_i_id, 'i')]:
                    if ('main_noun' in json_obj[id]
                            and json_obj[id]['main_noun'] is not None
                            and json_obj[id]['main_noun']['id'] in json_obj):
                        main_noun_id = json_obj[id]['main_noun']['id']
                        if 'modifiers' not in json_obj[main_noun_id]:
                            current_question_tags.append(
                                f"coordinated_noun_phrase:with_modifier_{prefix}:no"
                            )
                            continue
                        else:
                            current_question_tags.append(
                                f"coordinated_noun_phrase:with_modifier_{prefix}:yes"
                            )
                        if any([
                                modifier['type'] == 'prepositional_phrase' for
                                modifier in json_obj[main_noun_id]['modifiers']
                        ]):
                            current_question_tags.append(
                                f"coordinated_noun_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                            )
                        for clause in [
                                'relative_clause', 'reduced_relative_clause',
                                'wh_clause', 'that_clause',
                                'coordinated_subordinate_clause'
                        ]:
                            if any([
                                    clause in modifier['type'] for modifier in
                                    json_obj[main_noun_id]['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:with_{clause}"
                                )

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'noun_phrase': noun_phrase_i_text,
                        },
                        correct_answer=None,
                        question_tags=current_question_tags,
                        question_source=(node_id, noun_phrase_i_id),
                        node_id=node_id,
                        remove_phrases=coordinated_noun_phrase_texts,
                        coordinated_type='noun_phrase',
                        question_infos={
                            "knowledge_point": "coordinated_phrase"
                        }))

        return {"instances": instances}

    def generate_coordinated_noun_phrase_question_fitb(self, tree: nltk.Tree,
                                                       json_obj: dict):
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the noun phrase “{noun_phrase}” is coordinated with noun phrase ________."

        instances = []
        for node_id, node_info in json_obj.items():
            if 'coordinated_noun_phrases' not in node_info:
                continue
            coordinated_noun_phrases = node_info['coordinated_noun_phrases']
            coordinated_noun_phrase_texts = [
                coordinated_noun_phrase['plain_text']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is empty, skip
            if (None in coordinated_noun_phrase_texts
                    or '' in coordinated_noun_phrase_texts):
                continue
            coordinated_noun_phrase_ids = [
                coordinated_noun_phrase['id']
                for coordinated_noun_phrase in coordinated_noun_phrases
            ]
            # if any of the coordinated noun phrases is not in the json_obj, skip
            if any([
                    coordinated_noun_phrase_id not in json_obj for
                    coordinated_noun_phrase_id in coordinated_noun_phrase_ids
            ]):
                continue
            if len(set(coordinated_noun_phrase_texts)) < len(
                    coordinated_noun_phrase_texts):
                # the coordinated noun phrases are ambiguous
                continue
            if len(coordinated_noun_phrases) != 2:
                # there is not exactly two noun phrases, skip
                continue
            for i, coordinated_noun_phrase_i in enumerate(
                    coordinated_noun_phrases):
                for j, coordinated_noun_phrase_j in enumerate(
                        coordinated_noun_phrases):
                    if i == j:
                        continue
                    noun_phrase_i_text = coordinated_noun_phrase_i[
                        'plain_text'].strip().lower()
                    if self.noun_phrase_text_counter[noun_phrase_i_text] > 1:
                        # the noun phrase is ambiguous
                        continue
                    noun_phrase_j_text = coordinated_noun_phrase_j[
                        'plain_text'].strip().lower()
                    noun_phrase_i_id = coordinated_noun_phrase_i['id']
                    noun_phrase_j_id = coordinated_noun_phrase_j['id']
                    current_question_tags = ["coordinated_noun_phrase"]
                    with_modifier = {'i': 'no', 'j': 'no'}
                    for id, prefix in [(noun_phrase_i_id, 'i'),
                                       (noun_phrase_j_id, 'j')]:
                        if ('main_noun' in json_obj[id]
                                and json_obj[id]['main_noun'] is not None and
                                json_obj[id]['main_noun']['id'] in json_obj):
                            main_noun_id = json_obj[id]['main_noun']['id']
                            if 'modifiers' not in json_obj[main_noun_id]:
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:no"
                                )
                                continue
                            else:
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:yes"
                                )
                                with_modifier[prefix] = 'yes'
                            if any([
                                    modifier['type'] == 'prepositional_phrase'
                                    for modifier in json_obj[main_noun_id]
                                ['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_noun_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                                )
                            for clause in [
                                    'relative_clause',
                                    'reduced_relative_clause', 'wh_clause',
                                    'that_clause',
                                    'coordinated_subordinate_clause'
                            ]:
                                if any([
                                        clause in modifier['type'] for modifier
                                        in json_obj[main_noun_id]['modifiers']
                                ]):
                                    current_question_tags.append(
                                        f"coordinated_noun_phrase:with_modifier_{prefix}:with_{clause}"
                                    )

                    current_question_tags.append(
                        f"coordinated_noun_phrase:with_modifier:{with_modifier['i']}:{with_modifier['j']}"
                    )

                    instances.extend(
                        self.build_questions(
                            ingredient={
                                'noun_phrase': noun_phrase_i_text,
                            },
                            correct_answer=noun_phrase_j_text,
                            question_tags=current_question_tags,
                            question_source=(node_id, noun_phrase_i_id),
                            node_id=node_id,
                            remove_phrases=coordinated_noun_phrase_texts,
                            coordinated_type='noun_phrase',
                            question_infos={
                                "knowledge_point": "coordinated_phrase"
                            }))

        return {"instances": instances}

    def generate_coordinated_verb_phrase_question(self, tree: nltk.Tree,
                                                  json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, which of the following is coordinated with the verb phrase “{verb_phrase}”?",
        ]
        self.question_templates['yes_no'] = [
            "In the above sentence, the verb phrase “{verb_phrase}” is coordinated with verb phrase “{correct_answer}”.",
            "<NEG> In the above sentence, the verb phrase “{verb_phrase}” is not coordinated with verb phrase “{correct_answer}”.",
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            if 'predicates' not in node_info:
                continue
            coordinated_verb_phrases = node_info['predicates']
            coordinated_verb_phrase_texts = [
                verb_phrase['plain_text']
                for verb_phrase in coordinated_verb_phrases
            ]
            if len(coordinated_verb_phrases) < 2:
                # there is only one noun phrase, skip
                continue
            # if any of the coordinated verb phrases is empty, skip
            if (None in coordinated_verb_phrase_texts
                    or '' in coordinated_verb_phrase_texts):
                continue
            if len(set(coordinated_verb_phrase_texts)) < len(
                    coordinated_verb_phrase_texts):
                # the coordinated verb phrases are ambiguous
                continue
            if len(coordinated_verb_phrases) < 2:
                # there is only one noun phrase, skip
                continue
            for i, coordinated_verb_phrase_i in enumerate(
                    coordinated_verb_phrases):
                for j, coordinated_verb_phrase_j in enumerate(
                        coordinated_verb_phrases):
                    if i == j:
                        continue
                    verb_phrase_i_text = coordinated_verb_phrase_i[
                        'text'].strip().lower()
                    if self.verb_phrase_text_counter[verb_phrase_i_text] > 1:
                        # the verb phrase is ambiguous
                        continue
                    if verb_phrase_i_text in be_verbs:
                        continue
                    verb_phrase_j_text = coordinated_verb_phrase_j[
                        'text'].strip().lower()
                    verb_phrase_i_id = coordinated_verb_phrase_i['id']
                    verb_phrase_j_id = coordinated_verb_phrase_j['id']
                    current_question_tags = ["coordinated_verb_phrase"]
                    current_question_tags.extend([
                        # f"coordinated_verb_phrase:{json_obj[verb_phrase_i_id]['subtype']}:with:{json_obj[verb_phrase_j_id]['subtype']}",
                        f"coordinated_verb_phrase:with_modifier:{'yes' if 'modifiers' in json_obj[verb_phrase_i_id] else 'no'}:{'yes' if 'modifiers' in json_obj[verb_phrase_j_id] else 'no'}",
                        f"coordinated_verb_phrase:with_modifier_i:{'yes' if 'modifiers' in json_obj[verb_phrase_i_id] else 'no'}",
                        f"coordinated_verb_phrase:with_modifier_j:{'yes' if 'modifiers' in json_obj[verb_phrase_j_id] else 'no'}",
                    ])
                    for id, prefix in [(verb_phrase_i_id, 'i'),
                                       (verb_phrase_j_id, 'j')]:
                        if 'modifiers' in json_obj[id]:
                            if any([
                                    modifier['type'] == 'prepositional_phrase'
                                    for modifier in json_obj[id]['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_verb_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                                )
                            for clause in [
                                    'relative_clause',
                                    'reduced_relative_clause', 'wh_clause',
                                    'that_clause',
                                    'coordinated_subordinate_clause'
                            ]:
                                if any([
                                        clause in modifier['type'] for modifier
                                        in json_obj[id]['modifiers']
                                ]):
                                    current_question_tags.append(
                                        f"coordinated_verb_phrase:with_modifier_{prefix}:with_{clause}"
                                    )

                    instances.extend(
                        self.build_questions(
                            ingredient={
                                'verb_phrase': verb_phrase_i_text,
                            },
                            correct_answer=verb_phrase_j_text,
                            question_tags=current_question_tags,
                            question_source=(node_id, verb_phrase_i_id),
                            node_id=node_id,
                            remove_phrases=coordinated_verb_phrase_texts +
                            be_verbs,
                            coordinated_type='verb_phrase',
                            question_infos={
                                "knowledge_point": "coordinated_phrase"
                            }))

        return {"instances": instances}

    def generate_coordinated_verb_phrase_question_incorrect_answer_yn(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['yes_no'] = [
            "In the above sentence, the verb phrase “{verb_phrase}” is coordinated with verb phrase “{incorrect_answer}”.",
            "<NEG> In the above sentence, the verb phrase “{verb_phrase}” is not coordinated with verb phrase “{incorrect_answer}”.",
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            if 'predicates' not in node_info:
                continue
            coordinated_verb_phrases = node_info['predicates']
            coordinated_verb_phrase_texts = [
                verb_phrase['plain_text']
                for verb_phrase in coordinated_verb_phrases
            ]
            # if any of the coordinated verb phrases is empty, skip
            if (None in coordinated_verb_phrase_texts
                    or '' in coordinated_verb_phrase_texts):
                continue
            if len(set(coordinated_verb_phrase_texts)) < len(
                    coordinated_verb_phrase_texts):
                # the coordinated verb phrases are ambiguous
                continue
            for i, coordinated_verb_phrase_i in enumerate(
                    coordinated_verb_phrases):
                verb_phrase_i_text = coordinated_verb_phrase_i['text'].strip(
                ).lower()
                if self.verb_phrase_text_counter[verb_phrase_i_text] > 1:
                    # the verb phrase is ambiguous
                    continue
                if verb_phrase_i_text in be_verbs:
                    continue
                verb_phrase_i_id = coordinated_verb_phrase_i['id']
                current_question_tags = ["coordinated_verb_phrase"]
                current_question_tags.extend([
                    f"coordinated_verb_phrase:with_modifier_i:{'yes' if 'modifiers' in json_obj[verb_phrase_i_id] else 'no'}",
                ])
                for id, prefix in [(verb_phrase_i_id, 'i')]:
                    if 'modifiers' in json_obj[id]:
                        if any([
                                modifier['type'] == 'prepositional_phrase'
                                for modifier in json_obj[id]['modifiers']
                        ]):
                            current_question_tags.append(
                                f"coordinated_verb_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                            )
                        for clause in [
                                'relative_clause', 'reduced_relative_clause',
                                'wh_clause', 'that_clause',
                                'coordinated_subordinate_clause'
                        ]:
                            if any([
                                    clause in modifier['type']
                                    for modifier in json_obj[id]['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_verb_phrase:with_modifier_{prefix}:with_{clause}"
                                )

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'verb_phrase': verb_phrase_i_text,
                        },
                        correct_answer=None,
                        question_tags=current_question_tags,
                        question_source=(node_id, verb_phrase_i_id),
                        node_id=node_id,
                        remove_phrases=coordinated_verb_phrase_texts +
                        be_verbs,
                        coordinated_type='verb_phrase',
                        question_infos={
                            "knowledge_point": "coordinated_phrase"
                        }))

        return {"instances": instances}

    def generate_coordinated_verb_phrase_question_fitb(self, tree: nltk.Tree,
                                                       json_obj: dict):
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the verb phrase “{verb_phrase}” is coordinated with verb phrase ________."

        instances = []
        for node_id, node_info in json_obj.items():
            if 'predicates' not in node_info:
                continue
            coordinated_verb_phrases = node_info['predicates']
            coordinated_verb_phrase_texts = [
                verb_phrase['plain_text']
                for verb_phrase in coordinated_verb_phrases
            ]
            if len(coordinated_verb_phrases) < 2:
                # there is only one noun phrase, skip
                continue
            # if any of the coordinated verb phrases is empty, skip
            if (None in coordinated_verb_phrase_texts
                    or '' in coordinated_verb_phrase_texts):
                continue
            if len(set(coordinated_verb_phrase_texts)) < len(
                    coordinated_verb_phrase_texts):
                # the coordinated verb phrases are ambiguous
                continue
            if len(coordinated_verb_phrases) != 2:
                # there is only one noun phrase, skip
                continue
            for i, coordinated_verb_phrase_i in enumerate(
                    coordinated_verb_phrases):
                for j, coordinated_verb_phrase_j in enumerate(
                        coordinated_verb_phrases):
                    if i == j:
                        continue
                    verb_phrase_i_text = coordinated_verb_phrase_i[
                        'text'].strip().lower()
                    if self.verb_phrase_text_counter[verb_phrase_i_text] > 1:
                        # the verb phrase is ambiguous
                        continue
                    if verb_phrase_i_text in be_verbs:
                        continue
                    verb_phrase_j_text = coordinated_verb_phrase_j[
                        'text'].strip().lower()
                    verb_phrase_i_id = coordinated_verb_phrase_i['id']
                    verb_phrase_j_id = coordinated_verb_phrase_j['id']
                    current_question_tags = ["coordinated_verb_phrase"]
                    current_question_tags.extend([
                        # f"coordinated_verb_phrase:{json_obj[verb_phrase_i_id]['subtype']}:with:{json_obj[verb_phrase_j_id]['subtype']}",
                        f"coordinated_verb_phrase:with_modifier:{'yes' if 'modifiers' in json_obj[verb_phrase_i_id] else 'no'}:{'yes' if 'modifiers' in json_obj[verb_phrase_j_id] else 'no'}",
                        f"coordinated_verb_phrase:with_modifier_i:{'yes' if 'modifiers' in json_obj[verb_phrase_i_id] else 'no'}",
                        f"coordinated_verb_phrase:with_modifier_j:{'yes' if 'modifiers' in json_obj[verb_phrase_j_id] else 'no'}",
                    ])
                    for id, prefix in [(verb_phrase_i_id, 'i'),
                                       (verb_phrase_j_id, 'j')]:
                        if 'modifiers' in json_obj[id]:
                            if any([
                                    modifier['type'] == 'prepositional_phrase'
                                    for modifier in json_obj[id]['modifiers']
                            ]):
                                current_question_tags.append(
                                    f"coordinated_verb_phrase:with_modifier_{prefix}:with_prepositional_phrase"
                                )
                            for clause in [
                                    'relative_clause',
                                    'reduced_relative_clause', 'wh_clause',
                                    'that_clause',
                                    'coordinated_subordinate_clause'
                            ]:
                                if any([
                                        clause in modifier['type'] for modifier
                                        in json_obj[id]['modifiers']
                                ]):
                                    current_question_tags.append(
                                        f"coordinated_verb_phrase:with_modifier_{prefix}:with_{clause}"
                                    )

                    instances.extend(
                        self.build_questions(
                            ingredient={
                                'verb_phrase': verb_phrase_i_text,
                            },
                            correct_answer=verb_phrase_j_text,
                            question_tags=current_question_tags,
                            question_source=(node_id, verb_phrase_i_id),
                            node_id=node_id,
                            remove_phrases=coordinated_verb_phrase_texts +
                            be_verbs,
                            coordinated_type='verb_phrase',
                            question_infos={
                                "knowledge_point": "coordinated_phrase"
                            }))

        return {"instances": instances}
