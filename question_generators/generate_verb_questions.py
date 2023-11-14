from collections import Counter
import copy
import random
import nltk
from utils.extraction import QuestionGeneration
from utils.common import random_pronouns, be_verbs, random_verbs


class VerbQuestionGeneration(QuestionGeneration):
    skip_argument_types = {'empty'}

    def pre_process(self, *args, **kwds):
        args, kwds = super().pre_process(*args, **kwds)
        json_obj = kwds['json_obj']
        self.verb_phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if node_info['type'].startswith('predicate')
            and node_info['plain_text'] is not None
        ])
        self.pharses = list(self.verb_phrase_text_counter.keys())
        return args, kwds

    def get_argument_text(self,
                          argument_info,
                          json_obj,
                          tag_prefix='argument'):
        argument_type = argument_info['type']
        if argument_type == f'empty':
            argument_text = None
        else:
            argument_text = argument_info['plain_text'].strip()
            if argument_text == '':
                # still empty after rebuilding
                argument_text = None
        return argument_text, None

    def build_options(self, correct_answer, **kwds):
        options = [correct_answer]
        random_phrases = copy.deepcopy(self.pharses)
        random.shuffle(random_phrases)
        # remove the subject itself, its ancestors and its descendants
        for random_text in random_phrases:
            if random_text != '' and random_text not in options:
                options.append(random_text)
                if len(options) >= 4:
                    break

        shuffled_random_verbs = copy.deepcopy(random_verbs)
        random.shuffle(shuffled_random_verbs)
        for verb in shuffled_random_verbs:
            if len(options) >= 4:
                break
            if verb not in options:
                options.append(verb)

        return options

    def generate_verb_phrase_question_by_surface_subject(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, the grammatical subject of ____________ is “{grammatical_subject}”."
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, “{grammatical_subject}” is the grammatical subject of the verb phrase ____________."

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            question_tags.append(f"main_verb_phrase:surface_subject")
            if node_info['type'] != 'clause':
                continue
            if 'surface_subject' not in node_info:
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            surface_subject = node_info['surface_subject']
            surface_subject_type = surface_subject['type']
            if surface_subject_type in self.skip_argument_types:
                continue
            if surface_subject_type.startswith('unrecognized'):
                continue
            surface_subject_id = surface_subject['id']
            surface_subject_text, _ = self.get_argument_text(
                surface_subject, json_obj, tag_prefix='main_verb_phrase')
            if surface_subject_text is None or surface_subject_text == '':
                continue
            if surface_subject_id not in json_obj:
                continue
            if "(non-restrictive)" in json_obj[surface_subject_id]['type']:
                continue
            argument_json_obj = json_obj[surface_subject_id]
            if len([
                    x for x in (argument_json_obj.get('modifiers', []) +
                                argument_json_obj.get('complements', []) +
                                argument_json_obj.get('appositions', []))
                    if 'clause' in x['type']
            ]) > 0:
                # there are relative clauses of the noun phrase, ignore it
                continue
            surface_subject_text = surface_subject_text.strip()

            whole_sentence = self.whole_sentence.strip().lower()
            whole_sentence = whole_sentence.replace(
                surface_subject_text.lower(), '', 1).strip()
            if surface_subject_text.lower() in whole_sentence:
                # the subject is not unique
                continue

            predicates = node_info['predicates']
            if len(predicates) > 1:
                # ambiguous answer
                continue

            predicate = predicates[0]

            current_question_tags = copy.deepcopy(question_tags)
            if predicate['voice'] is None:
                continue
            current_question_tags.extend([
                f"main_verb_phrase:voice:{predicate['voice']}",
                f"main_verb_phrase:tense:{predicate['tense'][0]}:{predicate['tense'][1]}",
                f"main_verb_phrase:subtype:{predicate['subtype']}",
            ])

            verb_phrase_text = predicate['plain_text']
            if verb_phrase_text is None:
                continue
            verb_phrase_text = verb_phrase_text.strip().lower()
            if verb_phrase_text == '':
                continue

            if self.verb_phrase_text_counter[verb_phrase_text] > 1:
                # the noun phrase is ambiguous
                continue

            instances.extend(
                self.build_questions(
                    ingredient={
                        'grammatical_subject': surface_subject_text,
                    },
                    correct_answer=verb_phrase_text,
                    question_tags=current_question_tags,
                    question_source=(node_id, surface_subject_id),
                    node_id=node_id,
                    argument_id=surface_subject_id,
                    question_infos={"knowledge_point": "main_verb_phrase"}))

        return {"instances": instances}

    def generate_verb_phrase_question_by_direct_object(self, tree: nltk.Tree,
                                                       json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, the direct object of ____________ is “{direct_object}”."
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, “{direct_object}” is the direct object of the verb phrase ____________."

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            question_tags.append(f"main_verb_phrase:direct_object")
            if node_info['type'] != 'clause':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            if 'predicates' not in node_info:
                continue
            predicates = node_info['predicates']
            if len(predicates) < 1:
                continue
            for predicate in predicates:
                if predicate['voice'] == 'passive':
                    continue
                predicate_id = predicate['id']
                predicate_info = json_obj[predicate_id]
                if 'object' not in predicate_info:
                    continue
                direct_object = predicate_info['object']
                direct_object_type = direct_object['type']
                if direct_object_type in self.skip_argument_types:
                    continue
                if direct_object_type.startswith('unrecognized'):
                    continue
                direct_object_id = direct_object['id']
                direct_object_text, _ = self.get_argument_text(
                    direct_object, json_obj, tag_prefix='main_verb_phrase')
                if direct_object_text is None or direct_object_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend([
                    f"main_verb_phrase:voice:{predicate['voice']}",
                    f"main_verb_phrase:tense:{predicate['tense'][0]}:{predicate['tense'][1]}",
                    f"main_verb_phrase:subtype:{predicate['subtype']}",
                ])
                if direct_object_id not in json_obj:
                    continue
                if "(non-restrictive)" in json_obj[direct_object_id]['type']:
                    continue
                argument_json_obj = json_obj[direct_object_id]
                if len([
                        x for x in (argument_json_obj.get('modifiers', []) +
                                    argument_json_obj.get('complements', []) +
                                    argument_json_obj.get('appositions', []))
                        if 'clause' in x['type']
                ]) > 0:
                    # there are relative clauses of the noun phrase, ignore it
                    continue
                direct_object_text = direct_object_text.strip()

                whole_sentence = self.whole_sentence.strip().lower()
                whole_sentence = whole_sentence.replace(
                    direct_object_text.lower(), '', 1).strip()
                if direct_object_text.lower() in whole_sentence:
                    # the subject is not unique
                    continue

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip().lower()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text] > 1:
                    # the noun phrase is ambiguous
                    continue

                verb_phrase_words = verb_phrase_text.split()
                if verb_phrase_words[0] in be_verbs and verb_phrase_words[
                        -1] == 'going' or verb_phrase_words == ['used']:
                    if 'object' in predicate_info:
                        current_direct_object = predicate_info['object']
                        if current_direct_object[
                                'type'] == 'infinitive_clause':
                            # "be going to" or "used to", ignore them
                            continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'direct_object': direct_object_text,
                        },
                        correct_answer=verb_phrase_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, direct_object_id),
                        node_id=node_id,
                        argument_id=direct_object_id,
                        question_infos={"knowledge_point":
                                        "main_verb_phrase"}))

        return {"instances": instances}

    def generate_verb_phrase_question_by_subject_complement(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, “{subject_complement}” is the subject complement of the verb phrase ____________."
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, “{subject_complement}” is the subject complement of the verb phrase ____________."

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            question_tags += ['main_verb_phrase:subject_complement']
            if node_info['type'] != 'clause':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            subject_complement = []
            if 'predicates' not in node_info:
                continue
            predicates = node_info['predicates']
            if len(predicates) < 1:
                continue
            for predicate in predicates:
                if predicate['voice'] == 'passive':
                    continue
                predicate_id = predicate['id']
                predicate_info = json_obj[predicate_id]
                if 'subject_complement' not in predicate_info:
                    continue
                subject_complement = predicate_info['subject_complement']
                subject_complement_type = subject_complement['type']
                if subject_complement_type in self.skip_argument_types:
                    continue
                if subject_complement_type.startswith('unrecognized'):
                    continue
                subject_complement_id = subject_complement['id']
                subject_complement_text, _ = self.get_argument_text(
                    subject_complement,
                    json_obj,
                    tag_prefix='main_verb_phrase')
                if subject_complement_text is None or subject_complement_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend([
                    f"main_verb_phrase:voice:{predicate['voice']}",
                    f"main_verb_phrase:tense:{predicate['tense'][0]}:{predicate['tense'][1]}",
                    f"main_verb_phrase:subtype:{predicate['subtype']}",
                ])
                if subject_complement_id not in json_obj:
                    continue
                if "(non-restrictive)" in json_obj[subject_complement_id][
                        'type']:
                    continue
                argument_json_obj = json_obj[subject_complement_id]
                if len([
                        x for x in (argument_json_obj.get('modifiers', []) +
                                    argument_json_obj.get('complements', []) +
                                    argument_json_obj.get('appositions', []))
                        if 'clause' in x['type']
                ]) > 0:
                    # there are relative clauses of the noun phrase, ignore it
                    continue
                subject_complement_text = subject_complement_text.strip()

                whole_sentence = self.whole_sentence.strip().lower()
                whole_sentence = whole_sentence.replace(
                    subject_complement_text.lower(), '', 1).strip()
                if subject_complement_text.lower() in whole_sentence:
                    # the subject is not unique
                    continue

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip().lower()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'subject_complement': subject_complement_text,
                        },
                        correct_answer=verb_phrase_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, subject_complement_id),
                        node_id=node_id,
                        argument_id=subject_complement_id,
                        question_infos={"knowledge_point":
                                        "main_verb_phrase"}))

        return {"instances": instances}

    def generate_verb_phrase_question_by_indirect_object(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            "In the above sentence, the indirect object of ____________ is “{indirect_object}”."
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, “{indirect_object}” is the indirect object of the verb phrase ____________."

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            question_tags.append(f"main_verb_phrase:indirect_object")
            if node_info['type'] != 'clause':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            indirect_object = []
            if 'predicates' not in node_info:
                continue
            predicates = node_info['predicates']
            if len(predicates) < 1:
                continue
            for predicate in predicates:
                if predicate['voice'] == 'passive':
                    continue
                predicate_id = predicate['id']
                predicate_info = json_obj[predicate_id]
                if 'indirect_object' not in predicate_info:
                    continue
                indirect_object = predicate_info['indirect_object']
                indirect_object_type = indirect_object['type']
                if indirect_object_type in self.skip_argument_types:
                    continue
                if indirect_object_type.startswith('unrecognized'):
                    continue
                indirect_object_id = indirect_object['id']
                indirect_object_text, _ = self.get_argument_text(
                    indirect_object, json_obj, tag_prefix='main_verb_phrase')
                if indirect_object_text is None or indirect_object_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend([
                    f"main_verb_phrase:voice:{predicate['voice']}",
                    f"main_verb_phrase:tense:{predicate['tense'][0]}:{predicate['tense'][1]}",
                    f"main_verb_phrase:subtype:{predicate['subtype']}",
                ])
                if indirect_object_id not in json_obj:
                    continue
                if "(non-restrictive)" in json_obj[indirect_object_id]['type']:
                    continue
                argument_json_obj = json_obj[indirect_object_id]
                if len([
                        x for x in (argument_json_obj.get('modifiers', []) +
                                    argument_json_obj.get('complements', []) +
                                    argument_json_obj.get('appositions', []))
                        if 'clause' in x['type']
                ]) > 0:
                    # there are relative clauses of the noun phrase, ignore it
                    continue

                indirect_object_text = indirect_object_text.strip()

                whole_sentence = self.whole_sentence.strip().lower()
                whole_sentence = whole_sentence.replace(
                    indirect_object_text.lower(), '', 1).strip()
                if indirect_object_text.lower() in whole_sentence:
                    # the subject is not unique
                    continue

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip().lower()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'indirect_object': indirect_object_text,
                        },
                        correct_answer=verb_phrase_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, indirect_object_id),
                        node_id=node_id,
                        argument_id=indirect_object_id,
                        question_infos={"knowledge_point":
                                        "main_verb_phrase"}))

        return {"instances": instances}
