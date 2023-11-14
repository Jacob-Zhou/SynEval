from collections import Counter
import copy
import random
import nltk
import rich
from utils.extraction import QuestionGeneration
from utils.common import random_pronouns


class PrepositionalPhraseQuestionGeneration(QuestionGeneration):

    def pre_process(self, *args, **kwds):
        args, kwds = super().pre_process(*args, **kwds)
        json_obj = kwds['json_obj']
        self.pharses = [(node_id, node_info)
                        for node_id, node_info in json_obj.items()
                        if (node_info['type'].startswith('predicate')
                            or node_info['type'].startswith('noun_phrase'))]
        self.phrase_text_counter = Counter([
            node_info['plain_text'].lower().strip()
            for _, node_info in json_obj.items()
            if (node_info['type'].startswith('predicate')
                or node_info['type'].startswith('noun_phrase'))
            and node_info['plain_text'] is not None
        ])
        self.modifier_phrases = sum([
            node_info['modifiers']
            for _, node_info in json_obj.items() if 'modifiers' in node_info
        ], [])
        self.modifier_phrase_text_counter = Counter([
            self.get_modifier_text(node_info)[0].lower().strip()
            for node_info in self.modifier_phrases
            if (node_info['plain_text'] is not None
                and self.get_modifier_text(node_info)[0] is not None)
        ])
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

    def build_options(self, correct_answer, **kwds):
        options = [correct_answer]
        random_phrases = copy.deepcopy(self.pharses)
        random.shuffle(random_phrases)
        # remove the subject itself, its ancestors and its descendants
        for _, random_phrase in random_phrases:
            random_text = random_phrase['plain_text']
            if random_text is not None and random_text != '' and random_text not in options:
                if random_text in correct_answer:
                    # avoid the situation that, the correct answer is a phrase, and one of the options is the parent of the correct answer
                    # Question: In the above sentence, which following phrase is modified by the prepositional phrase “of the decision”?
                    # A. light
                    # B. light of the decision
                    continue
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

    def generate_prepositional_phrase_attachment_question(
            self, tree: nltk.Tree, json_obj: dict):

        # we don't generate fill-in-the-blank questions, since there may be multiple modifiers
        self.question_templates['multiple_choice'] = [
            'In the above sentence, which following phrase is modified by the prepositional phrase “{prepositional_phrase}”.'
            # 'In the above sentence, the prepositional phrase “{prepositional_phrase}” is modifying the phrase ________:'
        ]
        self.question_templates['fill_in_the_blank'] = [
            'In the above sentence, the prepositional phrase “{prepositional_phrase}” is modifying the phrase ________.'
        ]
        self.question_templates['yes_no'] = [
            'In the above sentence, the prepositional phrase “{prepositional_phrase}” is modifying the phrase “{correct_answer}”.',
            '<NEG> In the above sentence, the prepositional phrase “{prepositional_phrase}” is not modifying the phrase “{correct_answer}”.',
            'In the above sentence, the prepositional phrase “{prepositional_phrase}” is modifying the phrase “{incorrect_answer}”.',
            '<NEG> In the above sentence, the prepositional phrase “{prepositional_phrase}” is not modifying the phrase “{incorrect_answer}”.',
        ]

        instances = []
        for node_id, node_info in json_obj.items():

            question_tags = []
            if node_info['type'] not in {'predicate', 'noun_phrase'}:
                continue
            node_type = node_info['type']
            question_tags.append(f"subtype:{node_info['subtype']}")
            question_tags.append(
                f"pp_attachment:attach_to:{node_info['type']}")

            # mush have at least one modifier
            if 'modifiers' not in node_info:
                continue

            phrase_text = node_info['plain_text']
            if phrase_text is None:
                continue
            phrase_text = phrase_text.strip()
            if phrase_text == '':
                continue
            if self.phrase_text_counter[phrase_text.lower()] > 1:
                # the noun phrase is ambiguous
                continue

            modifier_infos = node_info.get('modifiers', [])
            if len(modifier_infos) < 1:
                continue

            for modifier_info in modifier_infos:
                modifier_text, modifier_question_tags = self.get_modifier_text(
                    modifier_info,
                    tag_prefix=('pp_attachment:adverbial_modifier'
                                if node_type == 'predicate' else
                                'pp_attachment:adjective_modifier'))
                if modifier_text is None:
                    continue
                if self.modifier_phrase_text_counter[
                        modifier_text.lower()] > 1:
                    # the modifier is ambiguous
                    continue
                if modifier_info['type'] != 'prepositional_phrase':
                    continue

                current_question_tags = copy.deepcopy(question_tags)
                current_question_tags.extend(modifier_question_tags)

                instances.extend(
                    self.build_questions(
                        ingredient={'prepositional_phrase': modifier_text},
                        correct_answer=phrase_text,
                        question_tags=current_question_tags,
                        question_source=(node_id, modifier_info['id']),
                        question_infos={
                            "knowledge_point":
                            "prepositional_phrase_attachment"
                        }))

        return {"instances": instances}
