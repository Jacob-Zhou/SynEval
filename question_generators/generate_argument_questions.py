from collections import Counter
import copy
import random
import nltk
from utils.extraction import QuestionGeneration
from utils.common import random_pronouns, be_verbs, random_verbs


class ArgumentQuestionGeneration(QuestionGeneration):
    skip_argument_types = {
        'existential_there', 'unrecognized', 'it-extraposition', 'it-cleft'
    }

    pharse_types = {'noun_phrase', 'adjective_phrase', 'adverb_phrase'}

    def pre_process(self, *args, **kwds):
        args, kwds = super().pre_process(*args, **kwds)
        json_obj = kwds['json_obj']
        self.pharses = [(node_id, node_info)
                        for node_id, node_info in json_obj.items()
                        if node_info['type'] in self.pharse_types]
        self.verb_phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if node_info['type'].startswith('predicate')
            and node_info['plain_text'] is not None
        ])
        return args, kwds

    def get_argument_text(self,
                          argument_info,
                          json_obj,
                          tag_prefix='argument'):
        argument_type = argument_info['type']
        question_tags = []
        if argument_type == f'empty':
            if argument_info['trace'] is not None:
                reference_index = str(argument_info['trace'])
                if reference_index in self.identity_brackets:
                    reference_node = self.identity_brackets[reference_index]
                    question_tags.append(f'{tag_prefix}:moved')
                    if reference_node['id'] in json_obj:
                        true_node = json_obj[reference_node['id']]
                        if true_node['type'] == 'complementizer':
                            question_tags.append(
                                f'{tag_prefix}:complementizer')
                            argument_text = true_node["specified_plain_text"]
                        else:
                            argument_text = true_node['plain_text']
                    else:
                        argument_text = reference_node['plain_text']
                else:
                    question_tags.append(f'{tag_prefix}:mistaken')
                    argument_text = None
            else:
                question_tags.append(f'{tag_prefix}:omitted')
                argument_text = 'omitted'
        else:
            question_tags.append(f'{tag_prefix}:{argument_type}')
            argument_text = argument_info['plain_text'].strip()
            if argument_text == '':
                argument_text = self.rebuild_text(argument_info['text'])
            if argument_text == '':
                # still empty after rebuilding
                argument_text = None
        return argument_text, question_tags

    def build_options(self, correct_answer, argument_id, **kwds):
        options = [correct_answer]
        random_phrases = copy.deepcopy(self.pharses)
        random.shuffle(random_phrases)
        # remove the subject itself, its ancestors and its descendants
        for random_phrase_id, random_phrase in random_phrases:
            if argument_id == random_phrase_id:
                continue
            if random_phrase_id[:-1].startswith(argument_id[:-1]):
                # the random phrase is a descendant of the subject
                continue
            if argument_id[:-1].startswith(random_phrase_id[:-1]):
                # the random phrase is an ancestor of the subject
                continue
            random_text = random_phrase['plain_text'].lower().strip()
            if random_text != '' and random_text not in options:
                options.append(random_text)
                if len(options) >= 4:
                    break

        shuffled_random_pronouns = copy.deepcopy(random_pronouns)
        random.shuffle(shuffled_random_pronouns)
        for pronoun in shuffled_random_pronouns:
            if len(options) >= 4:
                break
            if pronoun not in options:
                options.append(pronoun)

        return options

    def generate_surface_subject_question_by_verb_phrase(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            # "In the above sentence, the grammatical subject of “{verb_phrase}” is ____________:",
            "In the above sentence, which of the following is the grammatical subject of “{verb_phrase}”?"
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the grammatical subject of “{verb_phrase}” is ____________."
        self.question_templates['yes_no'] = [
            "In the above sentence, the grammatical subject of “{verb_phrase}” is “{correct_answer}”.",
            "<NEG> In the above sentence, the grammatical subject of “{verb_phrase}” is not “{correct_answer}”.",
            "In the above sentence, the grammatical subject of “{verb_phrase}” is “{incorrect_answer}”.",
            "<NEG> In the above sentence, the grammatical subject of “{verb_phrase}” is not “{incorrect_answer}”."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
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
            surface_subject_text, _question_tags = self.get_argument_text(
                surface_subject, json_obj, tag_prefix='surface_subject')
            if surface_subject_text is None or surface_subject_text == '':
                continue
            question_tags.extend(_question_tags)
            if surface_subject_id not in json_obj:
                continue
            if "(non-restrictive)" in json_obj[surface_subject_id]['type']:
                surface_subject_text = surface_subject_text.split(',')[0]
            surface_subject_text = surface_subject_text.strip().lower()
            for predicate in node_info['predicates']:
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.append(
                    f"surface_subject:voice:{predicate['voice']}")
                if predicate['voice'] is None:
                    continue
                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'verb_phrase': verb_phrase_text,
                        },
                        correct_answer=surface_subject_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, surface_subject_id),
                        node_id=node_id,
                        argument_id=surface_subject_id,
                        question_infos={"knowledge_point": "surface_subject"}))

        return {"instances": instances}

    def _generate_logical_subject_question_by_verb_phrase(
            self, tree: nltk.Tree, json_obj: dict):

        self.question_templates['multiple_choice'] = [
            # "In the above sentence, the logical subject of “{verb_phrase}” is ____________:",
            "In the above sentence, which of the following is the logical subject of “{verb_phrase}”?"
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the logical subject of “{verb_phrase}” is ____________."
        self.question_templates['yes_no'] = [
            "In the above sentence, the logical subject of “{verb_phrase}” is “{correct_answer}”.",
            "<NEG> In the above sentence, the logical subject of “{verb_phrase}” is not “{correct_answer}”.",
            "In the above sentence, the logical subject of “{verb_phrase}” is “{incorrect_answer}”.",
            "<NEG> In the above sentence, the logical subject of “{verb_phrase}” is not “{incorrect_answer}”."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            if node_info['type'] != 'clause':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            if 'predicates' not in node_info:
                continue
            predicates = node_info['predicates']
            if len(predicates) < 1:
                continue
            for predicate in predicates:
                if predicate['voice'] != 'passive':
                    continue
                predicate_id = predicate['id']
                predicate_info = json_obj[predicate_id]
                if 'logical_subjects' not in predicate_info:
                    continue
                logical_subjects = predicate_info['logical_subjects']
                if len(logical_subjects) != 1:
                    continue
                logical_subject = logical_subjects[0]
                logical_subject_type = logical_subject['type']
                if logical_subject_type in self.skip_argument_types:
                    continue
                if logical_subject_type.startswith('unrecognized'):
                    continue
                logical_subject_id = logical_subject['id']
                logical_subject_text, _question_tags = self.get_argument_text(
                    logical_subject, json_obj, tag_prefix='logical_subject')
                if logical_subject_text is None or logical_subject_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend(_question_tags)
                if "(non-restrictive)" in json_obj[logical_subject_id]['type']:
                    logical_subject_text = logical_subject_text.split(',')[0]
                logical_subject_text = logical_subject_text.strip().lower()

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'verb_phrase': verb_phrase_text,
                        },
                        correct_answer=logical_subject_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, logical_subject_id),
                        node_id=node_id,
                        argument_id=logical_subject_id,
                        question_infos={"knowledge_point": "logical_subject"}))

        return {"instances": instances}

    def generate_direct_object_question_by_verb_phrase(self, tree: nltk.Tree,
                                                       json_obj: dict):
        self.question_templates['multiple_choice'] = [
            # "In the above sentence, the direct object of “{verb_phrase}” is ____________:",
            "In the above sentence, which of the following is the direct object of “{verb_phrase}”?"
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the direct object of “{verb_phrase}” is ____________."
        self.question_templates['yes_no'] = [
            "In the above sentence, the direct object of “{verb_phrase}” is “{correct_answer}”.",
            "<NEG> In the above sentence, the direct object of “{verb_phrase}” is not “{correct_answer}”.",
            "In the above sentence, the direct object of “{verb_phrase}” is “{incorrect_answer}”.",
            "<NEG> In the above sentence, the direct object of “{verb_phrase}” is not “{incorrect_answer}”."
        ]
        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
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
                direct_object_text, _question_tags = self.get_argument_text(
                    direct_object, json_obj, tag_prefix='direct_object')
                if direct_object_text is None or direct_object_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend(_question_tags)
                if (direct_object_id in json_obj and "(non-restrictive)"
                        in json_obj[direct_object_id]['type']):
                    direct_object_text = direct_object_text.split(',')[0]
                direct_object_text = direct_object_text.strip().lower()

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
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
                            'verb_phrase': verb_phrase_text,
                        },
                        correct_answer=direct_object_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, direct_object_id),
                        node_id=node_id,
                        argument_id=direct_object_id,
                        question_infos={"knowledge_point": "direct_object"}))

        return {"instances": instances}

    def generate_subject_complement_question_by_verb_phrase(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            # "In the above sentence, the subject complement of “{verb_phrase}” is ____________:",
            "In the above sentence, which of the following is the subject complement of “{verb_phrase}”?"
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the subject complement of “{verb_phrase}” is ____________."
        self.question_templates['yes_no'] = [
            "In the above sentence, the subject complement of “{verb_phrase}” is “{correct_answer}”.",
            "<NEG> In the above sentence, the subject complement of “{verb_phrase}” is not “{correct_answer}”.",
            "In the above sentence, the subject complement of “{verb_phrase}” is “{incorrect_answer}”.",
            "<NEG> In the above sentence, the subject complement of “{verb_phrase}” is not “{incorrect_answer}”."
        ]
        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
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
                subject_complement_text, _question_tags = self.get_argument_text(
                    subject_complement,
                    json_obj,
                    tag_prefix='subject_complement')
                if subject_complement_text is None or subject_complement_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend(_question_tags)
                if (subject_complement_id in json_obj and "(non-restrictive)"
                        in json_obj[subject_complement_id]['type']):
                    subject_complement_text = subject_complement_text.split(
                        ',')[0]
                subject_complement_text = subject_complement_text.strip(
                ).lower()

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'verb_phrase': verb_phrase_text,
                        },
                        correct_answer=subject_complement_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, subject_complement_id),
                        node_id=node_id,
                        argument_id=subject_complement_id,
                        question_infos={
                            "knowledge_point": "subject_complement"
                        }))

        return {"instances": instances}

    def generate_indirect_object_question_by_verb_phrase(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['multiple_choice'] = [
            # "In the above sentence, the indirect object of “{verb_phrase}” is ____________:",
            "In the above sentence, which of the following is the indirect object of “{verb_phrase}”?"
        ]
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, the indirect object of “{verb_phrase}” is ____________."
        self.question_templates['yes_no'] = [
            "In the above sentence, the indirect object of “{verb_phrase}” is “{correct_answer}”.",
            "<NEG> In the above sentence, the indirect object of “{verb_phrase}” is not “{correct_answer}”.",
            "In the above sentence, the indirect object of “{verb_phrase}” is “{incorrect_answer}”.",
            "<NEG> In the above sentence, the indirect object of “{verb_phrase}” is not “{incorrect_answer}”."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
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
                indirect_object_text, _question_tags = self.get_argument_text(
                    indirect_object, json_obj, tag_prefix='indirect_object')
                if indirect_object_text is None or indirect_object_text == '':
                    continue
                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend(_question_tags)
                if "(non-restrictive)" in json_obj[indirect_object_id]['type']:
                    indirect_object_text = indirect_object_text.split(',')[0]
                indirect_object_text = indirect_object_text.strip().lower()

                verb_phrase_text = predicate['plain_text']
                if verb_phrase_text is None:
                    continue
                verb_phrase_text = verb_phrase_text.strip()
                if verb_phrase_text == '':
                    continue
                if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                    # the noun phrase is ambiguous
                    continue

                instances.extend(
                    self.build_questions(
                        ingredient={
                            'verb_phrase': verb_phrase_text,
                        },
                        correct_answer=indirect_object_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, indirect_object_id),
                        node_id=node_id,
                        argument_id=indirect_object_id,
                        question_infos={"knowledge_point": "indirect_object"}))

        return {"instances": instances}
