from itertools import groupby
from utils.common import be_verbs, have_verbs, auxiliary_verbs, modal_verbs
import re
import nltk

reBULLSHIT = re.compile(
    r"^(?:.*\s)?(?:(?:is)|(?:the phrase)|(?:the words?)) [\"“](.*?)[\"”].*$")
reCPANSWER = re.compile(
    # r"^(?:.*\s)?(?:is coordinated with the) (?:(?:noun)|(?:verb)) (?:phrase) [\"“](.*?)[\"”].*$"
    r"^(?:.*\s)?(?:(?:noun)|(?:verb)) (?:phrase) [\"“](.*?)[\"”] (?:is coordinated with the) (?:(?:noun)|(?:verb)) (?:phrase) [\"“](.*?)[\"”].*$"
)
reMVPANSWER = re.compile(
    r"^(?:.*\s)?(?:the (?:(?:direct object)|(?:indirect object)|(?:grammatical subject)|(?:subject complement)) of(?: the verb phrase)?) [\"“](.*?)[\"”] is [\"“](.*?)[\"”].*$"
)
reMVPANSWER_2 = re.compile(
    r"^(?:.*?\s)?[\"“](.*?)[\"”] is (?:the (?:(?:direct object)|(?:indirect object)|(?:grammatical subject)|(?:subject complement)) of(?: the verb phrase)?) [\"“](.*?)[\"”]\.?$"
)
reCLOSED_BRACKET = re.compile(r"\((.*?)\)")
reCLOSED_QUATION = re.compile(r" '(.*?)'")
reMC = [
    re.compile(r"^(?:Answer: ?)?([A-D])[^a-zA-Z]?$"),  # A
    re.compile(r"^(?:Answer: ?)?([A-D])[ .,)].*$"),  # A.
    re.compile(
        r"^(?:.*\s)?(?:answer is) \\*[\"“]?([A-D])[.,]?(?:\\*[\"” ].*)?")
]
reYN = [
    re.compile(r"^(?:Answer: ?)?(?:([Yy]es|[Nn]o))[^a-zA-Z]?$"),  # A
    re.compile(r"^(?:Answer: ?)?(?:([Yy]es|[Nn]o))[ .,].*$"),  # A.
    re.compile(
        r"^(?:.*\s)?(?:answer is) \\*[\"“]?(?:([Yy]es|[Nn]o))[.,]?(?:\\*[\"” ].*)?$"
    ),
    re.compile(r"^(?:Answer: ?)?(?:([Tt]rue|[Ff]alse))[^a-zA-Z]?$"),  # A
    re.compile(r"^(?:Answer: ?)?(?:([Tt]rue|[Ff]alse))[ .,].*$"),  # A.
    re.compile(
        r"^(?:.*\s)?(?:answer is) \\*[\"“]?(?:([Tt]rue|[Ff]alse))[.,]?(?:\\*[\"” ].*)?$"
    ),
]


def tokenize(text):
    return nltk.word_tokenize(text.lower())


# "yes_no", "fill_in_the_blank", "short_answer", "multiple_choice"


def get_candidate_answers(tokenized):
    if len(tokenized) <= 1:
        return [tokenized]
    else:
        if tokenized[0].lower() in {"a", "an", "the"}:
            return [tokenized, tokenized[1:]]
        elif tokenized[0].lower() in set(be_verbs + have_verbs +
                                         auxiliary_verbs + modal_verbs +
                                         ['to']):
            return ([tokenized, tokenized[1:]] +
                    get_candidate_answers(tokenized[1:]))
        else:
            return [tokenized]


def get_possible_answers(text, question_type, knowledge_point=None):
    text = text.strip()
    text = text.split("\n")[0]
    if question_type in {"fill_in_the_blank", "short_answer"}:
        text = text.lower()

        answer = [text.strip()]
        any_match = False
        if match := reMVPANSWER.match(text):
            any_match = True
            if knowledge_point == "main_verb_phrase":
                answer.append(match.group(1).strip())
            else:
                answer.append(match.group(2).strip())
        elif (match := reMVPANSWER_2.match(text)):
            any_match = True
            if knowledge_point == "main_verb_phrase":
                answer.append(match.group(2).strip())
            else:
                answer.append(match.group(1).strip())
        elif match := reBULLSHIT.match(text):
            any_match = True
            answer.append(match.group(1).strip())
        if match := reCPANSWER.match(text):
            any_match = True
            answer.append(match.group(1).strip())
            answer.append(match.group(2).strip())

        if not any_match:
            if "there is no" in text:
                answer = ["<no-answer>"]
            else:
                answer = [text.strip()]

        answer.append(reCLOSED_QUATION.sub(' " \\1 "', answer[0]).strip())
        # remove brackets
        answer.append(reCLOSED_BRACKET.sub('', answer[0]).strip())
        return list(sorted(set(answer)))
    elif question_type == "yes_no":
        for regex in reYN:
            match = regex.match(text)
            if match:
                return [match.group(1).strip()]
    elif question_type == "multiple_choice":
        for regex in reMC:
            match = regex.match(text)
            if match:
                return [match.group(1).strip()]
    else:
        raise NotImplementedError()


class Edit:

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A token span edit list: [o_start, o_end, c_start, c_end]
    # Input 4: An error type string, if known
    def __init__(self, orig, cor, edit, type="NA"):
        # Orig offsets, spacy tokens and string
        self.o_start = edit[0]
        self.o_end = edit[1]
        self.o_toks = orig[self.o_start:self.o_end]
        self.o_str = " ".join(self.o_toks) if self.o_toks else ""
        # Cor offsets, spacy tokens and string
        self.c_start = edit[2]
        self.c_end = edit[3]
        self.c_toks = cor[self.c_start:self.c_end]
        self.c_str = " ".join(self.c_toks) if self.c_toks else ""
        # Error type
        self.type = type

    # Minimise the edit; e.g. [a b -> a c] = [b -> c]
    def minimise(self):
        # While the first token is the same on both sides
        while self.o_toks and self.c_toks and \
                self.o_toks[0] == self.c_toks[0]:
            # Remove that token from the span, and adjust the start offsets
            self.o_toks = self.o_toks[1:]
            self.c_toks = self.c_toks[1:]
            self.o_start += 1
            self.c_start += 1
        # Do the same for the last token
        while self.o_toks and self.c_toks and \
                self.o_toks[-1] == self.c_toks[-1]:
            self.o_toks = self.o_toks[:-1]
            self.c_toks = self.c_toks[:-1]
            self.o_end -= 1
            self.c_end -= 1
        # Update the strings
        self.o_str = " ".join(self.o_toks) if self.o_toks else ""
        self.c_str = " ".join(self.c_toks) if self.c_toks else ""
        return self

    # Input: An id for the annotator
    # Output: An edit string formatted for an M2 file
    def to_m2(self, id=0):
        span = " ".join(["A", str(self.o_start), str(self.o_end)])
        cor_toks_str = " ".join([tok for tok in self.c_toks])
        return "|||".join(
            [span, self.type, cor_toks_str, "REQUIRED", "-NONE-",
             str(id)])

    # Edit object string representation
    def __str__(self):
        orig = "Orig: " + str([self.o_start, self.o_end, self.o_str])
        cor = "Cor: " + str([self.c_start, self.c_end, self.c_str])
        type = "Type: " + repr(self.type)
        return ", ".join([orig, cor, type])


class Alignment:

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    def __init__(self, orig, cor, lev=False):
        # Set orig and cor
        self.orig = orig
        self.cor = cor
        # Align orig and cor and get the cost and op matrices
        self.cost_matrix, self.op_matrix = self.align(lev)
        # Get the cheapest align sequence from the op matrix
        self.align_seq = self.get_cheapest_align_seq()

    # Input: A flag for standard Levenshtein alignment
    # Output: The cost matrix and the operation matrix of the alignment
    def align(self, lev):
        # Sentence lengths
        o_len = len(self.orig)
        c_len = len(self.cor)
        # Lower case token IDs (for transpositions)
        o_low = [o.lower() for o in self.orig]
        c_low = [c.lower() for c in self.cor]
        # Create the cost_matrix and the op_matrix
        cost_matrix = [[0.0 for j in range(c_len + 1)]
                       for i in range(o_len + 1)]
        op_matrix = [["O" for j in range(c_len + 1)] for i in range(o_len + 1)]
        # Fill in the edges
        for i in range(1, o_len + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            op_matrix[i][0] = "D"
        for j in range(1, c_len + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            op_matrix[0][j] = "I"

        # Loop through the cost_matrix
        for i in range(o_len):
            for j in range(c_len):
                # Matches
                if self.orig[i] == self.cor[j]:
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    op_matrix[i + 1][j + 1] = "M"
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + 1
                    ins_cost = cost_matrix[i + 1][j] + 1
                    trans_cost = float("inf")
                    # Standard Levenshtein (S = 1)
                    if lev:
                        sub_cost = cost_matrix[i][j] + 1
                        # Linguistic Damerau-Levenshtein
                    else:
                        # Custom substitution
                        sub_cost = cost_matrix[i][j] + \
                            self.get_sub_cost(self.orig[i], self.cor[j])
                        # Transpositions require >=2 tokens
                        # Traverse the diagonal while there is not a Match.
                        k = 1
                        while i-k >= 0 and j-k >= 0 and \
                                cost_matrix[i-k+1][j-k+1] != cost_matrix[i-k][j-k]:
                            if sorted(o_low[i - k:i + 1]) == sorted(
                                    c_low[j - k:j + 1]):
                                trans_cost = cost_matrix[i - k][j - k] + k
                                break
                            k += 1
                    # Costs
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    l = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i + 1][j + 1] = costs[l]
                    if l == 0: op_matrix[i + 1][j + 1] = "T" + str(k + 1)
                    elif l == 1: op_matrix[i + 1][j + 1] = "S"
                    elif l == 2: op_matrix[i + 1][j + 1] = "I"
                    else: op_matrix[i + 1][j + 1] = "D"
        # Return the matrices
        return cost_matrix, op_matrix

    # Input 1: A spacy orig Token
    # Input 2: A spacy cor Token
    # Output: A linguistic cost between 0 < x < 2
    def get_sub_cost(self, o, c):
        # Short circuit if the only difference is case
        if o.lower() == c.lower(): return 0
        # Char cost
        char_cost = Indel.normalized_distance(o, c)
        # Combine the costs
        return char_cost

    # Get the cheapest alignment sequence and indices from the op matrix
    # align_seq = [(op, o_start, o_end, c_start, c_end), ...]
    def get_cheapest_align_seq(self):
        i = len(self.op_matrix) - 1
        j = len(self.op_matrix[0]) - 1
        align_seq = []
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = self.op_matrix[i][j]
            # Matches and substitutions
            if op in {"M", "S"}:
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            # Deletions
            elif op == "D":
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            # Insertions
            elif op == "I":
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            # Transpositions
            else:
                # Get the size of the transposition
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

    # all-split: Don't merge anything
    def get_all_split_edits(self):
        edits = []
        for align in self.align_seq:
            edits.append(Edit(self.orig, self.cor, align[1:]))
        return edits

    # all-equal: Merge all edits of the same operation type.
    def get_all_equal_edits(self):
        edits = []
        for op, group in groupby(self.align_seq, lambda x: x[0]):
            merged = self.merge_edits(list(group))
            edits.append(Edit(self.orig, self.cor, merged[0][1:], op))
        return edits

    # Merge the input alignment sequence to a single edit span
    def merge_edits(self, seq):
        if seq: return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
        else: return seq

    # Alignment object string representation
    def __str__(self):
        orig = " ".join(["Orig:"] + [tok for tok in self.orig])
        cor = " ".join(["Cor:"] + [tok for tok in self.cor])
        cost_matrix = "\n".join(["Cost Matrix:"] +
                                [str(row) for row in self.cost_matrix])
        op_matrix = "\n".join(["Operation Matrix:"] +
                              [str(row) for row in self.op_matrix])
        seq = "Best alignment: " + str([a[0] for a in self.align_seq])
        return "\n".join([orig, cor, cost_matrix, op_matrix, seq])


def calc_f1(hyp, ref, punct_weight=0.1):
    hyp = tokenize(hyp)
    ref = tokenize(ref)
    alignment = Alignment(hyp, ref)
    merged_alignment = alignment.get_all_equal_edits()
    # Calculate precision, recall and f1
    tp = 0
    p_count = 0
    g_count = 0
    for chunk in merged_alignment:
        if chunk.type == 'M':
            # equal
            tp += 1
            p_count += 1
            g_count += 1
        elif chunk.type in {'S', 'T'}:
            # substitution
            if chunk.o_str in punctuation:
                p_count += 1 * punct_weight
            else:
                p_count += 1
            if chunk.c_str in punctuation:
                g_count += 1 * punct_weight
            else:
                g_count += 1
        elif chunk.type == 'I':
            # insertion
            if chunk.c_str in punctuation:
                g_count += 1 * punct_weight
            else:
                g_count += len(chunk.c_str.split())
        elif chunk.type == 'D':
            # deletion
            if chunk.o_str in punctuation:
                p_count += 1 * punct_weight
            else:
                p_count += len(chunk.o_str.split())
    return tp, p_count, g_count