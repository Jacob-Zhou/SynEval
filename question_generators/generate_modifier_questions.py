from collections import Counter
import copy
import random
import nltk
import rich
from utils.extraction import QuestionGeneration
from utils.common import random_postmodifiers
from utils.fn import Label


class ModifierQuestionGeneration(QuestionGeneration):

    pharse_types = {'noun_phrase', 'adjective_phrase', 'adverb_phrase'}

    def pre_process(self, *args, **kwds):
        args, kwds = super().pre_process(*args, **kwds)
        json_obj = kwds['json_obj']
        self.noun_phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if node_info['type'].startswith('noun_phrase')
        ])
        self.verb_phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if node_info['type'].startswith('predicate')
            and node_info['plain_text'] is not None
        ])
        self.modifier_phases = sum([
            node_info['modifiers']
            for _, node_info in json_obj.items() if 'modifiers' in node_info
        ], [])
        return args, kwds

    def get_modifier_text(self, modifier_info, tag_prefix='argument'):
        try:
            modifier_type = modifier_info['type']
        except KeyError:
            rich.print(modifier_info)
            raise Exception(f"modifier_info does not have type")
        question_tags = [f"{tag_prefix}:{modifier_type}"]
        modifier_text = modifier_info['plain_text'].strip()
        if modifier_text == '':
            modifier_text = self.rebuild_text(modifier_info['text'])
            question_tags.append(f'{tag_prefix}:moved')
        if modifier_text == '':
            # still empty after rebuilding
            return None, question_tags
        else:
            return modifier_text, question_tags

    def build_options(self, correct_answer, remove_from_options, **kwds):
        options = [correct_answer]
        random_phrases = copy.deepcopy(self.modifier_phases)
        random.shuffle(random_phrases)
        # remove the subject itself, its ancestors and its descendants
        for random_phrase in random_phrases:
            random_text = self.get_modifier_text(random_phrase)[0]
            if random_text is not None and random_text != '' and random_text not in options:
                if (random_text not in remove_from_options
                        and random_text not in {'n\'t', 'not'}):
                    options.append(random_text)
                if len(options) >= 4:
                    break

        shuffled_random_postmodifiers = copy.deepcopy(random_postmodifiers)
        random.shuffle(shuffled_random_postmodifiers)
        for postmodifier in shuffled_random_postmodifiers:
            if len(options) >= 4:
                break
            if postmodifier not in options:
                options.append(postmodifier)

        return options

    def generate_adjective_modifier_existent_question(self, tree: nltk.Tree,
                                                      json_obj: dict):
        self.question_templates['yes_no'] = [
            "In the above sentence, the noun phrase “{correct_answer}” has at least one post-modifier or complement.",
            "<NEG> In the above sentence, the noun phrase “{correct_answer}” does not have any post-modifier or complement."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            if node_info['type'] != 'noun_phrase':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            question_tags.append('existent')
            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue
            modifiers = node_info['modifiers']
            if len(modifiers) < 1:
                continue
            modifiers = [
                modifier for modifier in modifiers
                if ('unrecognized' not in modifier['type']
                    and modifier['plain_text'].strip() != ''
                    and not modifier['id'][:-1].startswith(node_id[:-1]))
            ]
            if len(modifiers) < 1:
                continue
            noun_phrase_text = node_info['plain_text'].strip()
            if noun_phrase_text == '':
                continue
            if self.noun_phrase_text_counter[noun_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue
            instances.extend(
                self.build_questions(
                    ingredient={},
                    correct_answer=noun_phrase_text,
                    question_tags=question_tags,
                    question_source=(node_id, ),
                    question_infos={"knowledge_point": "adjective_modifier"}))

        return {"instances": instances}

    def generate_adjective_modifier_not_existent_question(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['yes_no'] = [
            "<NEG> In the above sentence, the noun phrase “{correct_answer}” has at least one post-modifier or complement.",
            "In the above sentence, the noun phrase “{correct_answer}” does not have any post-modifier or complement."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            if node_info['type'] != 'noun_phrase':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            question_tags.append('not-existent')
            # build position to tree from node_id
            position_to_tree = list(
                map(int,
                    node_id.split('-')[-1][1:-1].split('.')))
            parent_node = tree[position_to_tree[:-1]]
            if parent_node.label() != '' and Label(
                    parent_node.label()).tag == 'NP':
                # may have a modifier, not safe to generate this question
                continue
            self_node = tree[position_to_tree]
            children = {Label(child.label()).tag for child in self_node}
            if not children.isdisjoint({'NP', 'S', 'SBAR'}):
                # have noun phrase child or have clausal complement
                continue
            noun_phrase_text = node_info['plain_text'].strip()
            if noun_phrase_text == '':
                continue
            if self.noun_phrase_text_counter[noun_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue
            instances.extend(
                self.build_questions(
                    ingredient={},
                    correct_answer=noun_phrase_text,
                    question_tags=question_tags,
                    question_source=(node_id, ),
                    question_infos={"knowledge_point": "adjective_modifier"}))

        return {"instances": instances}

    def generate_adjective_modifier_question(self, tree: nltk.Tree,
                                             json_obj: dict):

        # we don't generate fill-in-the-blank questions, since there may be multiple modifiers
        self.question_templates[
            'multiple_choice'] = "In the above sentence, which of the following is a post-modifier or complement of the noun phrase “{noun_phrase}”?"
        self.question_templates['yes_no'] = [
            "In the above sentence, “{correct_answer}” is a post-modifier or complement of the noun phrase “{noun_phrase}”.",
            "In the above sentence, “{incorrect_answer}” is a post-modifier or complement of the noun phrase “{noun_phrase}”."
        ]

        instances = []
        for node_id, node_info in json_obj.items():

            question_tags = []
            if node_info['type'] != 'noun_phrase':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")

            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue

            noun_phrase_text = node_info['plain_text'].strip()
            if noun_phrase_text == '':
                continue
            if self.noun_phrase_text_counter[noun_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue

            modifier_infos = node_info.get('modifiers', [])
            if len(modifier_infos) < 1:
                continue

            remove_from_options = set()
            modifiers = []

            for modifier_info in modifier_infos:
                if 'unrecognized' in modifier_info['type']:
                    continue

                if modifier_info['id'][:-1].startswith(node_id[:-1]):
                    # the modifier is the noun phrase itself or its ancestor
                    continue

                modifier_text, modifier_question_tags = self.get_modifier_text(
                    modifier_info, tag_prefix='adjective_modifier')
                if modifier_text is None:
                    continue
                remove_from_options.add(modifier_text)

                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.append(modifier_info['type'])
                current_question_tags.extend(modifier_question_tags)

                modifiers.append((modifier_text, current_question_tags))

            for modifier_text, question_tags in modifiers:
                instances.extend(
                    self.build_questions(
                        ingredient={'noun_phrase': noun_phrase_text},
                        correct_answer=modifier_text,
                        question_tags=question_tags,
                        question_source=(node_id, modifier_info['id']),
                        remove_from_options=remove_from_options,
                        question_infos={
                            "knowledge_point": "adjective_modifier"
                        }))

        return {"instances": instances}

    def generate_adjective_modifier_fitb_question(self, tree: nltk.Tree,
                                                  json_obj: dict):

        # we don't generate fill-in-the-blank questions, since there may be multiple modifiers
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, there is a post-modifier or complement of the noun phrase “{noun_phrase}”. It is ________."

        instances = []
        for node_id, node_info in json_obj.items():

            question_tags = []
            if node_info['type'] != 'noun_phrase':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")

            # mush have at least one modifier
            if 'modifiers' not in node_info and 'complements' not in node_info:
                continue

            noun_phrase_text = node_info['plain_text'].strip()
            if noun_phrase_text == '':
                continue
            if self.noun_phrase_text_counter[noun_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue

            modifier_infos_map = {
                'modifiers': node_info.get('modifiers', []),
                'complements': node_info.get('complements', [])
            }

            remove_from_options = set()
            all_modifiers = []
            safe_modifiers = []

            for modifier_type, modifier_infos in modifier_infos_map.items():
                for modifier_info in modifier_infos:
                    is_safe = modifier_type == 'modifiers'
                    if 'unrecognized' in modifier_info['type']:
                        is_safe = False

                    if modifier_info['id'][:-1].startswith(node_id[:-1]):
                        # the modifier is the noun phrase itself or its ancestor
                        is_safe = False

                    modifier_text, modifier_question_tags = self.get_modifier_text(
                        modifier_info, tag_prefix='adjective_modifier')
                    if modifier_text is None:
                        is_safe = False

                    remove_from_options.add(modifier_text)

                    current_question_tags = copy.deepcopy(question_tags)
                    current_question_tags.append(modifier_info['type'])
                    current_question_tags.extend(modifier_question_tags)

                    modifier_tuple = (modifier_text, current_question_tags)
                    all_modifiers.append(modifier_tuple)
                    if is_safe:
                        safe_modifiers.append(modifier_tuple)

            if len(all_modifiers) > 1 or len(safe_modifiers) != 1:
                continue

            modifier_text, question_tags = safe_modifiers[0]
            instances.extend(
                self.build_questions(
                    ingredient={'noun_phrase': noun_phrase_text},
                    correct_answer=modifier_text,
                    question_tags=question_tags,
                    question_source=(node_id, modifier_info['id']),
                    remove_from_options=remove_from_options,
                    question_infos={"knowledge_point": "adjective_modifier"}))

        return {"instances": instances}

    def generate_adverbial_modifier_existent_question(self, tree: nltk.Tree,
                                                      json_obj: dict):
        self.question_templates['yes_no'] = [
            "In the above sentence, the verb phrase “{correct_answer}” has at least one adverbial modifier.",
            "<NEG> In the above sentence, the verb phrase “{correct_answer}” does not have any adverbial modifier."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            if node_info['type'] != 'predicate':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            question_tags.append('existent')
            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue
            modifiers = node_info['modifiers']
            if len(modifiers) < 1:
                continue

            modifiers = [
                modifier for modifier in modifiers
                if ('unrecognized' not in modifier['type']
                    and modifier['plain_text'].strip() != ''
                    and not modifier['id'][:-1].startswith(node_id[:-1]))
            ]

            if len(modifiers) < 1:
                continue
            # use `text' instead of `plain_text', because `plain_text' includes part of abverbial modifier
            verb_phrase_text = node_info['text']
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
                    ingredient={},
                    correct_answer=verb_phrase_text,
                    question_tags=question_tags,
                    question_source=(node_id, ),
                    question_infos={"knowledge_point": "adverbial_modifier"}))

        return {"instances": instances}

    def generate_adverbial_modifier_not_existent_question(
            self, tree: nltk.Tree, json_obj: dict):
        self.question_templates['yes_no'] = [
            "<NEG> In the above sentence, the verb phrase “{correct_answer}” has at least one adverbial modifier.",
            "In the above sentence, the verb phrase “{correct_answer}” does not have any adverbial modifier."
        ]

        instances = []
        for node_id, node_info in json_obj.items():
            question_tags = []
            if node_info['type'] != 'predicate':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")
            question_tags.append('not-existent')
            # build position to tree from node_id
            # mush have at least one modifier
            if 'modifiers' in node_info:
                continue
            # use `text' instead of `plain_text', because `plain_text' includes part of abverbial modifier
            verb_phrase_text = node_info['text']
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
                    ingredient={},
                    correct_answer=verb_phrase_text,
                    question_tags=question_tags,
                    question_source=(node_id, ),
                    question_infos={"knowledge_point": "adverbial_modifier"}))

        return {"instances": instances}

    def generate_adverbial_modifier_question(self, tree: nltk.Tree,
                                             json_obj: dict):

        # we don't generate fill-in-the-blank questions, since there may be multiple modifiers
        self.question_templates[
            'multiple_choice'] = "In the above sentence, which of the following is an adverbial modifier of the verb phrase “{verb_phrase}”?"
        self.question_templates['yes_no'] = [
            "In the above sentence, “{correct_answer}” is an adverbial modifier of the verb phrase “{verb_phrase}”.",
            "In the above sentence, “{incorrect_answer}” is an adverbial modifier of the verb phrase “{verb_phrase}”."
        ]

        instances = []
        for node_id, node_info in json_obj.items():

            question_tags = []
            if node_info['type'] != 'predicate':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")

            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue

            # use `text' instead of `plain_text', because `plain_text' includes part of abverbial modifier
            verb_phrase_text = node_info['text']
            if verb_phrase_text is None:
                continue
            verb_phrase_text = verb_phrase_text.strip()
            if verb_phrase_text == '':
                continue
            if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue

            modifier_infos = node_info.get('modifiers', [])
            if len(modifier_infos) < 1:
                continue

            remove_from_options = set()
            modifiers = []

            for modifier_info in modifier_infos:
                if 'unrecognized' in modifier_info['type']:
                    continue

                if modifier_info['id'][:-1].startswith(node_id[:-1]):
                    # the modifier is the verb phrase itself or its ancestor
                    continue

                modifier_text, modifier_question_tags = self.get_modifier_text(
                    modifier_info, tag_prefix='adverbial_modifier')
                if modifier_text is None:
                    continue
                remove_from_options.add(modifier_text)

                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.append(modifier_info['type'])
                current_question_tags.extend(modifier_question_tags)

                modifiers.append((modifier_text, current_question_tags))

            for modifier_text, question_tags in modifiers:
                instances.extend(
                    self.build_questions(
                        ingredient={'verb_phrase': verb_phrase_text},
                        correct_answer=modifier_text,
                        question_tags=question_tags,
                        question_source=(node_id, modifier_info['id']),
                        remove_from_options=remove_from_options,
                        question_infos={
                            "knowledge_point": "adverbial_modifier"
                        }))

        return {"instances": instances}

    def generate_adverbial_modifier_fitb_question(self, tree: nltk.Tree,
                                                  json_obj: dict):

        # we don't generate fill-in-the-blank questions, since there may be multiple modifiers
        self.question_templates[
            'fill_in_the_blank'] = "In the above sentence, there is an adverbial modifier of the verb phrase “{verb_phrase}”. It is ________."

        instances = []
        for node_id, node_info in json_obj.items():

            question_tags = []
            if node_info['type'] != 'predicate':
                continue
            question_tags.append(f"subtype:{node_info['subtype']}")

            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue

            # use `text' instead of `plain_text', because `plain_text' includes part of abverbial modifier
            verb_phrase_text = node_info['text']
            if verb_phrase_text is None:
                continue
            verb_phrase_text = verb_phrase_text.strip()
            if verb_phrase_text == '':
                continue
            if self.verb_phrase_text_counter[verb_phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue

            modifier_infos = node_info.get('modifiers', [])
            if len(modifier_infos) < 1:
                continue

            remove_from_options = set()
            modifiers = []
            safe_modifiers = []

            for modifier_info in modifier_infos:
                is_safe = True
                if 'unrecognized' in modifier_info['type']:
                    is_safe = False

                if modifier_info['id'][:-1].startswith(node_id[:-1]):
                    # the modifier is the verb phrase itself or its ancestor
                    is_safe = False

                modifier_text, modifier_question_tags = self.get_modifier_text(
                    modifier_info, tag_prefix='adverbial_modifier')
                if modifier_text is None:
                    is_safe = False
                remove_from_options.add(modifier_text)

                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.append(modifier_info['type'])
                current_question_tags.extend(modifier_question_tags)

                modifier_tuple = (modifier_text, current_question_tags)
                if is_safe:
                    safe_modifiers.append(modifier_tuple)

            if len(modifiers) > 1 or len(safe_modifiers) != 1:
                continue

            modifier_text, question_tags = safe_modifiers[0]

            instances.extend(
                self.build_questions(
                    ingredient={'verb_phrase': verb_phrase_text},
                    correct_answer=modifier_text,
                    question_tags=question_tags,
                    question_source=(node_id, modifier_info['id']),
                    remove_from_options=remove_from_options,
                    question_infos={"knowledge_point": "adverbial_modifier"}))

        return {"instances": instances}
