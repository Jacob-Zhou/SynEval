import unittest
import nltk
import rich
from unittest import TestCase

from utils.fn import get_verb_type_from_verb_sequence, get_verb_type_from_tree


class TestVerbType(unittest.TestCase):

    def test_simple_present(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBP file) (NP (PRP$ their) (NNS reports)) (ADVP-TMP (RB late)))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1, [('normal', ('present', None), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (VBZ does) (RB n't) (VP (VB have) (S (NP-SBJ (-NONE- *-1)) (VP (TO to) (VP (VB play) (NP (DT the) (JJ same) (NNP Mozart) (CC and) (NNP Strauss) (NNS concertos)) (ADVP (RB over) (CC and) (RB over)) (ADVP-TMP (RB again)))))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2, [('normal', ('present', None), 'active')])

        tree3 = nltk.Tree.fromstring(
            "(VP (VBZ is) (VP (VP (VBN purged) (NP (-NONE- *-1)) (PP (IN of) (NP (JJ threatening) (NNS elements)))) (, ,) (VP (VBN served) (NP (-NONE- *-1)) (PRT (RP up)) (PP (IN in) (NP (JJ bite-sized) (NNS morsels)))) (CC and) (VP (VBN accompanied) (NP (-NONE- *-1)) (PP (IN by) (NP-LGS (NNS visuals))))))"
        )
        tree_type3 = get_verb_type_from_tree(tree3)
        self.assertEqual(tree_type3,
                         [('normal', ('present', None), 'passive'),
                          ('normal', ('present', None), 'passive'),
                          ('normal', ('present', None), 'passive')])

        tree4 = nltk.Tree.fromstring(
            "(VP (ADVP-MNR (RB clearly)) (VBZ demonstrates) (SBAR (IN that) (S (NP-SBJ (NNP Mips)) (VP (VBZ is) (NP-PRD (NP (DT a) (NNS systems) (NN company)) (CONJP (RB rather) (IN than)) (NP (RB just) (DT a) (NN chip) (NN company)))))))"
        )
        tree_type4 = get_verb_type_from_tree(tree4)
        self.assertEqual(tree_type4, [('normal', ('present', None), 'active')])

    def test_simple_past(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBD was) (VP (ADVP-MNR (RB mistakenly)) (VBN attributed) (NP (-NONE- *-2)) (PP-CLR (TO to) (NP (NNP Christina) (NNP Haag)))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1, [('normal', ('past', None), 'passive')])

    def test_simple_future(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (MD will) (VP (VB be) (ADJP-PRD (JJ payable) (NP-TMP (NNP Feb.) (CD 15)))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1, [('normal', ('future', None), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (MD shall) (VP (VB be) (VP (VBN based) (NP (-NONE- *-45)) (PP-LOC-CLR (IN in) (NP (NNP Indianapolis))))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2, [('normal', ('future', None), 'passive')])

    def test_simple_future_in_past(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (MD would) (VP (VB be) (ADJP-PRD (JJ payable) (NP-TMP (NNP Feb.) (CD 15)))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('normal', ('future_in_past', None), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (MD would) (VP (VB be) (VP (VBN based) (NP (-NONE- *-45)) (PP-LOC-CLR (IN in) (NP (NNP Indianapolis))))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2,
                         [('normal', ('future_in_past', None), 'passive')])

    def test_perfect(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBZ has) (ADVP (RB historically)) (VP (VBN paid) (NP (NN obeisance)) (PP-CLR (TO to) (NP (NP (DT the) (NN ideal)) (PP (IN of) (NP (DT a) (JJ level) (NN playing) (NN field)))))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('normal', ('present', 'perfect'), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (VBD had) (ADVP (RB just)) (VP (VBN been) (VP (VBN released) (NP (-NONE- *-9)))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2,
                         [('normal', ('past', 'perfect'), 'passive')])

    def test_continuous(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBP are) (VP (VBG appealing) (PP-CLR (TO to) (NP-1 (DT the) (NNPS Securities) (CC and) (NNP Exchange) (NNP Commission))) (S-CLR (NP-SBJ (-NONE- *-1)) (RB not) (VP (TO to) (VP (VB limit) (NP (NP (PRP$ their) (NN access)) (PP (TO to) (NP (NP (NN information)) (PP (IN about) (NP (NP (NN stock) (NNS purchases) (CC and) (NNS sales)) (PP (IN by) (NP (JJ corporate) (NNS insiders)))))))))))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('normal', ('present', 'continuous'), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (MD will) (VP (VB be) (VP (VBG going) (PP-CLR (IN for) (NP (DT a) (JJ full) (NN bid))))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2,
                         [('normal', ('future', 'continuous'), 'active')])

        tree3 = nltk.Tree.fromstring(
            "(VP (VBZ is) (VP (VBG being) (VP (VBN acquired) (NP (-NONE- *-35)) (PP (IN by) (NP-LGS (NP (NNP Sony) (NNP Corp.)) (, ,) (SBAR (WHNP-1 (WDT which)) (S (NP-SBJ-36 (-NONE- *T*-1)) (VP (VBZ is) (VP (VBN based) (NP (-NONE- *-36)) (PP-LOC-CLR (IN in) (NP (NNP Japan))))))))))))"
        )
        tree_type3 = get_verb_type_from_tree(tree3)
        self.assertEqual(tree_type3,
                         [('normal', ('present', 'continuous'), 'passive')])

    def test_perfect_continuous(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBD had) (VP (VBN been) (VP (VBG casting) (ADVP-LOC (RB abroad)) (S-PRP (NP-SBJ (-NONE- *-1)) (VP (TO to) (VP (VB raise) (NP (NP (NP (NP (DT the) (QP (CD 3) (CD billion)) (NNP New) (NNP Zealand) (NNS dollars)) (PRN (-LRB- -LRB-) (NP (QP ($ US$) (CD 1.76) (CD billion)) (-NONE- *U*)) (-RRB- -RRB-))) (TO to) (NP (QP ($ NZ$) (CD 4) (CD billion)) (-NONE- *U*))) (SBAR (WHNP-2 (-NONE- 0)) (S (NP-SBJ-5 (PRP it)) (VP (VBZ needs) (S (NP-SBJ (-NONE- *-5)) (VP (TO to) (VP (VB come) (ADVP-CLR (RB up)) (PP-CLR (IN with) (NP (-NONE- *T*-2))) (PP-TMP (IN by) (NP (NP (DT the) (NN end)) (PP (IN of) (NP (PRP$ its) (JJ fiscal) (NN year))) (NP-TMP (IN next) (NNP June) (CD 30)))))))))))))))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('normal',
                           ('past', 'perfect_continuous'), 'active')])

    def test_present_participle(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBG proclaiming) (S (NP-SBJ (PRP him)) (`` ``) (NP-PRD (NP (DT the) (JJ great) (NN improviser)) (PP (IN of) (NP (DT the) (JJ 18th) (NN century))))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('present_participle', (None, None), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (VBG being) (VP (VBN sold) (CC or) (VBN closed) (NP (-NONE- *-1))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2,
                         [('present_participle', (None, None), 'passive'),
                          ('present_participle', (None, None), 'passive')])

    def test_perfect_participle(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBG having) (VP (VBN met) (ADVP-TMP (NP (QP (IN about) (CD 25)) (NNS years)) (RB ago))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('perfect_participle', (None, 'perfect'), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (VBG having) (VP (VBN been) (VP (VBN set) (NP (-NONE- *-1)))))"
        )
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2, [('perfect_participle',
                                       (None, 'perfect'), 'passive')])

    def test_past_participle(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (VBN discovered) (NP (-NONE- *)) (PP-TMP (IN during) (NP (DT the) (JJ past) (CD three) (NNS years))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        self.assertEqual(tree_type1,
                         [('past_participle', (None, None), 'passive')])

    def test_infinitive(self):
        tree1 = nltk.Tree.fromstring(
            "(VP (TO to) (VP (VB ease) (NP (NP (NN reporting) (NNS requirements)) (PP (IN for) (NP (DT some) (NN company) (NNS executives))))))"
        )
        tree_type1 = get_verb_type_from_tree(tree1)
        rich.print(tree_type1)
        self.assertEqual(tree_type1, [('infinitive', (None, None), 'active')])

        tree2 = nltk.Tree.fromstring(
            "(VP (TO to) (VP (VB be) (VP (VBN believed) (NP (-NONE- *-11)))))")
        tree_type2 = get_verb_type_from_tree(tree2)
        self.assertEqual(tree_type2, [('infinitive', (None, None), 'passive')])

        tree3 = nltk.Tree.fromstring(
            "(VP (TO to) (VP (VB have) (VP (VBN referred) (PP-CLR (TO to) (NP (NP (DT a) (NN letter)) (SBAR (SBAR (WHNP-1 (IN that)) (S (NP-SBJ (PRP he)) (VP (VBD said) (SBAR (-NONE- 0) (S (NP-SBJ (NNP President) (NNP Bush)) (VP (VBD sent) (NP (-NONE- *T*-1)) (PP-DTV (TO to) (NP (JJ Colombian) (NNP President) (NNP Virgilio) (NNP Barco))))))))) (, ,) (CC and) (SBAR (WHPP-3 (IN in) (WHNP (WDT which))) (S (NP-SBJ (NNP President) (NNP Bush)) (VP (VBD said) (PP-LOC (-NONE- *T*-3)) (SBAR (-NONE- 0) (S (NP-SBJ (NP (PRP it)) (S (-NONE- *EXP*-4))) (VP (VBD was) (ADJP-PRD (JJ possible)) (S-4 (NP-SBJ (-NONE- *)) (VP (TO to) (VP (VB overcome) (NP (NP (NNS obstacles)) (PP (TO to) (NP (DT a) (JJ new) (NN agreement)))))))))))))))))))"
        )
        tree_type3 = get_verb_type_from_tree(tree3)
        self.assertEqual(tree_type3,
                         [('infinitive', (None, 'perfect'), 'active')])

        tree4 = nltk.Tree.fromstring(
            "(VP (TO to) (VP (VB be) (VP (VBG considering) (S (NP-SBJ (-NONE- *-2)) (VP (VBG getting) (PP-CLR (IN into) (NP (DT the) (NN auto-making) (NN business))))))))"
        )
        tree_type4 = get_verb_type_from_tree(tree4)
        self.assertEqual(tree_type4,
                         [('infinitive', (None, 'continuous'), 'active')])


if __name__ == '__main__':
    unittest.main()