import sys
import math
from Transition import Transition
from collections import defaultdict
from helper_functions import Helper as h


def dot(features, weights):
    score = 0.0
    for key in set(features) & set(weights):
        score += features[key] * weights[key]
    return score

class PerceptronModel:
    LEX_FEAT = h.LEX_FEAT
    POS_FEAT = h.POS_FEAT
    DEP_FEAT = h.DEP_FEAT

    STACK_SOURCE = h.STACK_SOURCE
    BUFF_SOURCE = h.BUFF_SOURCE
    INPUT_SOURCE = h.INPUT_SOURCE

    def __init__(self, labeled):
        self.labeled = labeled
        self.learning_rate = 1.0
        self.weights = defaultdict(float)
        self.label_set = ('abbrev acomp advcl advmod amod appos ' +  \
            'attr aux auxpass cc ccomp complm conj cop csubj ' + \
            'dep det dobj expl infmod iobj mark measure neg ' +  \
            'nn nsubj nsubjpass null num number parataxis partmod ' + \
            'pcomp pobj poss possessive preconj pred predet ' +  \
            'prep prt punct purpcl quantmod rcmod rel tmod ' +   \
            'xcomp').split() if labeled else [None]

    def try_get_token(self, source, index):
        try:
            return(source[index])
        except:
            return(None)

    def get_source(self, stack, buff, input_sentence, source_type):
        source = None
        if source_type == self.STACK_SOURCE:
            source = stack
        elif source_type == self.BUFF_SOURCE:
            source = buff
        elif source_type == self.INPUT_SOURCE:
            source = input_sentence[::-1]
        else:
            print >>sys.stderr, 'Invalid source type'
        return(source)

    def get_input_offset_token(self, token, input_sentence, input_offset):
        if h.get_id(token) + input_offset <= 0: #To prevent wraparound
            return(None)
        return(self.try_get_token(input_sentence, -(h.get_id(token) + input_offset))) # No +1 since the token id starts from 1 instead of 0

    def get_head_offset_token(self, token, input_sentence, head_multiplier, arcs):
        while (token is not None and head_multiplier > 0):
            head_multiplier -= 1
            token_id = h.get_id(token)
            head_id = arcs.get(token_id, None)
            if head_id == None:
                token = None
            else:
                token = self.try_get_token(input_sentence, -head_id)
        return(token)

    def get_all_children(self, token, input_sentence, arcs):
        all_children = {}
        head_id = h.get_id(token)
        for tail in arcs.keys():
            if arcs[tail] == head_id:
                child_token = self.try_get_token(input_sentence, -(tail))
                if child_token is not None:
                    all_children[h.get_id(child_token)] = child_token
                else:
                    print >>sys.stderr, 'Non-existent child, should NOT happen!!!'
        return(all_children)


    def get_leftmost_child(self, token, input_sentence, arcs, leftmost_multiplier):
        assert leftmost_multiplier < 0, "Invalid leftmost_multiplier passed"
        while token is not None and leftmost_multiplier < 0:
            leftmost_multiplier += 1
            all_children = self.get_all_children(token, input_sentence, arcs)
            if len(all_children) == 0 :
                return(None)
            min_candidate = min(all_children.keys())
            if (min_candidate < h.get_id(token)):
                token = all_children[min_candidate]
            else:
                return(None)
        return(token)


    def get_rightmost_child(self, token, input_sentence, arcs, rightmost_multiplier):
        assert rightmost_multiplier > 0, "Invalid rightmost_multiplier passed"
        while token is not None and rightmost_multiplier > 0:
            rightmost_multiplier -= 1
            all_children = self.get_all_children(token, input_sentence, arcs)
            if len(all_children) == 0 :
                return(None)
            max_candidate = max(all_children.keys())
            if (max_candidate > h.get_id(token)):
                token = all_children[max_candidate]
            else:
                return(None)
        return(token)

    def get_left_rightmost_child(self, token, input_sentence, arcs, left_rightmost_multiplier):
        if left_rightmost_multiplier == 0:
            return(token)
        elif left_rightmost_multiplier < 0:
            return(self.get_leftmost_child(token, input_sentence, arcs, left_rightmost_multiplier))
        else:
            return(self.get_rightmost_child(token, input_sentence, arcs, left_rightmost_multiplier))

    def get_all_siblings(self, token, input_sentence, arcs):
        if arcs.get(h.get_id(token), None) is None:
            return([])
        all_siblings = []
        for word in input_sentence:
            if h.get_id(word) != 0 and h.get_id(word) != h.get_id(token) and arcs.get(h.get_id(word), None) is not None and arcs[h.get_id(word)] == arcs[h.get_id(token)]:
                all_siblings += [word]
        return(all_siblings)

    def get_left_sibling(self, token, input_sentence, arcs, left_sibling_multiplier):
        assert left_sibling_multiplier < 0, "Invalid left sibling multiplier"
        all_siblings = self.get_all_siblings(token, input_sentence, arcs)
        while(token is not None and left_sibling_multiplier < 0):
            left_sibling_multiplier += 1
            if len(all_siblings) == 0:
                return(None)
            min_dist = h.INFINITY
            nearest_sibling = None
            for sibling in all_siblings:
                if h.get_id(sibling) < h.get_id(token) and abs(h.get_id(sibling) - h.get_id(token)) < min_dist:
                    min_dist = abs(h.get_id(sibling) - h.get_id(token))
                    nearest_sibling = sibling
            token = nearest_sibling
            # NOTE: It is NOT possible that we keep cycling between siblings.
        return(token)

    def get_right_sibling(self, token, input_sentence, arcs, right_sibling_multiplier):
        assert right_sibling_multiplier > 0, "Invalid right sibling multiplier"
        all_siblings = self.get_all_siblings(token, input_sentence, arcs)
        while(token is not None and right_sibling_multiplier > 0):
            right_sibling_multiplier -= 1
            if len(all_siblings) == 0:
                return(None)
            min_dist = h.INFINITY
            nearest_sibling = None
            for sibling in all_siblings:
                if h.get_id(sibling) > h.get_id(token) and abs(h.get_id(sibling) - h.get_id(token)) < min_dist:
                    min_dist = abs(h.get_id(sibling) - h.get_id(token))
                    nearest_sibling = sibling
            token = nearest_sibling
            # NOTE: It is possible NOT that we keep cycling between siblings.
        return(token)

    def get_left_right_sibling(self, token, input_sentence, arcs, left_right_sibling_multiplier):
        if left_right_sibling_multiplier == 0:
            return(token)
        elif left_right_sibling_multiplier < 0:
            return(self.get_left_sibling(token, input_sentence, arcs, left_right_sibling_multiplier))
        else:
            return(self.get_right_sibling(token, input_sentence, arcs, left_right_sibling_multiplier))


    def get_model7_params(self, stack, buff, input_sentence, arcs, labels, tType, feat_type, source_type, source_offset = 0, input_offset = 0, head_multiplier = 0, left_rightmost_multiplier = 0, left_right_sibling_specifier = 0, suffix_len = 0): #Described here: http://stp.lingfil.uu.se/~nivre/docs/maltparser.pdf
        assert feat_type in [self.DEP_FEAT, self.POS_FEAT, self.LEX_FEAT], "Invalid feat_type specified"
        assert source_type in [self.BUFF_SOURCE, self.STACK_SOURCE, self.INPUT_SOURCE], "Invalid source type specified"
        assert source_offset >= 0, "Invalid source_offset"
        assert head_multiplier >= 0, "Invalid head multiplier specified"
        rev_input_sentence = input_sentence[::-1]
        source = self.get_source(stack, buff, input_sentence, source_type)#Reverse the input sentence if it isn't already reversed by get_source

        token = self.try_get_token(source, -(source_offset + 1))
        if token is None:
            return(None)
        if input_offset != 0:
            token = self.get_input_offset_token(token, rev_input_sentence, input_offset)
        if token is None:
            return(None)
        token = self.get_head_offset_token(token, rev_input_sentence, head_multiplier, arcs)
        if token is None:
            return(None)
        token = self.get_left_rightmost_child(token, rev_input_sentence, arcs, left_rightmost_multiplier)
        if token is None:
            return(None)
        token = self.get_left_right_sibling(token, rev_input_sentence, arcs, left_right_sibling_specifier)
        if token is None:
            return(None)
        ret_str = 'transition=%d,feat_type=%d,source_type=%d,source_offset=%d,input_offset=%d,head_multiplier=%d,left_rightmost_multiplier=%d,left_right_sibling_specifier=%d' % (tType, feat_type, source_type, source_offset, input_offset, head_multiplier , left_rightmost_multiplier, left_right_sibling_specifier)
        if feat_type == self.LEX_FEAT:
            #suffix len can be specified via argument
            lex_feat = h.get_word(token)
            if suffix_len > 0:
                ret_str += 'lex_feat=%s' %(lex_feat[-suffix_len:])
            else:
                ret_str += 'lex_feat=%s' %(lex_feat)
        elif feat_type == self.DEP_FEAT:
            dep_feat = labels.get(h.get_id(token), None)
            if  dep_feat is not None:
                ret_str += 'dep_feat=%s' %(dep_feat)
            else:
                return(None)
        elif feat_type == self.POS_FEAT:
            pos_feat = h.get_postag(token)
            ret_str += 'pos_feat=%s' % (pos_feat)
        else:
            return(None)
        return(ret_str)

    def get_valency(self, arcs, head):
        left_valency = 0
        right_valency = 0
        for tail in arcs.keys():
            if arcs.get(tail, -1) == head:
                if tail < head:
                    left_valency += 1
                elif tail > head:
                    right_valency += 1
        return([left_valency, right_valency])

    def compose_feats(self, features, feat_str_list):
        feat_str = ''
        for feat in feat_str_list:
            if feat is None:
                return(None)
            feat_str += feat
        features[feat_str] = 1
        return(feat_str)


    def extract_features(self, transition, stack, buff, labels, previous_transitions, arcs, input_sentence):
        features = defaultdict(float)
        tType = transition.transitionType
        label = transition.label

        #Model7 features as described in http://stp.lingfil.uu.se/~nivre/docs/maltparser.pdf

        feat1_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 1)#dep for pre-top
        feat2_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 1, 0, 0, -1) #dep for pre-top's lmc
        feat21_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 1, 0, 0, -1) #pos for pre-top's lmc
        feat3_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 1, 0, 0, 1)#dep or pre-top's rmc
        feat31_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 1, 0, 0, 1)#pos or pre-top's rmc
        feat4_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 0, 0, 0, -1)#dep for top's lmc
        feat41_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 0, 0, 0, -1)#pos for top's lmc
        feat5_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 1)#lex for pre-top
        feat6_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 0)#lex for top

        feat7_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.BUFF_SOURCE)#lex for next buffer item
        feat71_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.BUFF_SOURCE)#pos for next buffer item

        feat75_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.BUFF_SOURCE, 1)#lex for next-next buffer item
        feat76_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.BUFF_SOURCE, 1)#pos for next-next buffer item

        feat8_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 1, 1)#lex for word after pre-top in input
        feat9_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 1, -1)#lex for word before pre-top in input

        feat10_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 1, 1)#pos for word after pre-top in input
        feat11_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 1, -1)#pos for word before pre-top in input

        feat12_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 1, 1)#dep for word after pre-top in input
        feat13_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 1, -1)#dep for word before pre-top in input

        feat14_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 0, 1)#lex for word after top in input
        feat15_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.LEX_FEAT, self.STACK_SOURCE, 0, -1)#lex for word before top in input

        feat16_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 0, 1)#pos for word after top in input
        feat17_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.POS_FEAT, self.STACK_SOURCE, 0, -1)#pos for word before top in input

        feat18_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 0, 1)#dep for word after top in input
        feat19_model7 = self.add_model7_feat(features, stack, buff, input_sentence, arcs, labels, tType,self.DEP_FEAT, self.STACK_SOURCE, 0, -1)#dep for word before top in input

        pre_top_pos = self.get_model7_params(stack, buff, input_sentence, arcs, labels, tType, self.POS_FEAT, self.STACK_SOURCE, 1)
        top_pos = self.get_model7_params(stack, buff, input_sentence, arcs, labels, tType, self.POS_FEAT, self.STACK_SOURCE, 0)
        cfeat_1 = self.compose_feats(features, [feat5_model7, pre_top_pos])#lex_pos of pre-top
        cfeat_2 = self.compose_feats(features, [feat6_model7, top_pos])#lex_pos for top

        cfeat_3 = self.compose_feats(features, [feat7_model7, feat71_model7])#lex_pos for next buffer item
        cfeat_4 = self.compose_feats(features, [feat75_model7,feat76_model7])#lex_pos for next-next buffer item

        cfeat_5 = self.compose_feats(features, [cfeat_1, cfeat_2])#lex_pos for both pre-top and top
        cfeat_6 = self.compose_feats(features, [cfeat_1, feat6_model7])#lex_pos of pre-top with lex of top
        cfeat_7 = self.compose_feats(features, [feat5_model7, cfeat_2])#lex of pre-top with lex_pos of top
        cfeat_8 = self.compose_feats(features, [cfeat_1, top_pos])#lex_pos of pre-top with pos of top
        cfeat_9 = self.compose_feats(features, [pre_top_pos, cfeat_2]) #pos of pre-top with lex_pos of top
        cfeat_10 = self.compose_feats(features, [feat5_model7, feat6_model7])#lex of both pre_top and top
        cfeat_11 = self.compose_feats(features, [pre_top_pos, top_pos])#pos of both pre_top and top
        cfeat_12 = self.compose_feats(features, [top_pos, feat71_model7])#pos of top and next buff
        cfeat_13 = self.compose_feats(features, [top_pos, feat71_model7, feat76_model7])#pos for top next and next next
        cfeat_14 = self.compose_feats(features, [pre_top_pos, top_pos, feat71_model7])#pos for pre-top, top and next
        cfeat_15 = self.compose_feats(features, [pre_top_pos, feat21_model7, top_pos])#pos for pre-top pre top lmc and top
        cfeat_16 = self.compose_feats(features, [pre_top_pos, feat31_model7, top_pos])#pos for pre-top, pre-top rmc and top
        cfeat_17 = self.compose_feats(features, [pre_top_pos, top_pos, feat41_model7])#pos for pre-top, top and top's lmc


        # Top two POS tags from the stack
        for i in range(3):#was originally 2
            if i >= len(stack):
                break
            s = stack[-(i+1)]
            pos = s[3]
            features['transition=%d,s%d.pos=%s' % (tType, i, pos)] = 1

        # Next four POS tags from the buffer
        for i in range(3):
            if i >= len(buff):
                break
            b = buff[-(i+1)]
            pos = b[3]
            features['transition=%d,b%d.pos=%s' % (tType, i, pos)] = 1

        # Previous transition type
        if len(previous_transitions) > 0:
            prev = previous_transitions[-1]
            features['transition=%d,prev_transition=%d' % (tType, prev)] = 1
        else:
            features['transition=%d,prev_transition=None' % (tType)] = 1

        # Bias feature
        features['transition=%d' % (transition.transitionType)] = 1

        if self.labeled:
            # Action and label pair
            features['transition=%d,label=%s' % (transition.transitionType, transition.label)] = 1
            # Label bias
            features['label=%s' % (transition.label)] = 1

        #Features based on http://dl.acm.org/citation.cfm?id=2002777
        #Distance function
        if len(stack) > 0 and len(buff) > 0:
            dist = h.get_id(stack[-1]) - h.get_id(buff[-1])
            if dist < 0:
                features['transition=%d,neg_dist=' % (tType)] = dist
            else:
                features['transition=%d,pos_dist=' % (tType)] = dist

        #Valency function
        if len(stack) > 1:
            if tType == Transition.LeftArc: # Left Arc
                [left_valency, right_valency] = self.get_valency(arcs, h.get_id(stack[-1]))
                features['transition=%d,head_left_valency=%d' % (tType, left_valency)] = 1
                features['transition=%d,head_right_valency=%d' % (tType, right_valency)] = 1
            elif tType == Transition.RightArc:#should probably check for right arc here!
                [left_valency, right_valency] = self.get_valency(arcs, h.get_id(stack[-2]))
                features['transition=%d,head_left_valency=%d' % (tType, left_valency)] = 1
                features['transition=%d,head_right_valency=%d' % (tType, right_valency)] = 1
        return features

    def add_model7_feat(self, feature_dict, stack, buff, input_sentence, arcs, labels, tType, feat_type, source_type, source_offset = 0, input_offset = 0, head_multiplier = 0, left_rightmost_multiplier = 0, left_right_sibling_specifier = 0, suffix_len = 0):
        new_feat = self.get_model7_params(stack, buff, input_sentence, arcs, labels, tType, feat_type, source_type, source_offset, input_offset, head_multiplier, left_rightmost_multiplier, left_right_sibling_specifier, suffix_len)
        if new_feat is not None:
            feature_dict[new_feat] = 1
        return(new_feat)

    def possible_transitions(self, stack, buff):
        possible_transitions = []
        if len(buff) >= 1:
            possible_transitions.append(Transition(Transition.Shift, None))
        if len(stack) >= 2:
            for label in self.label_set:
                possible_transitions.append(Transition(Transition.LeftArc, label))
                possible_transitions.append(Transition(Transition.RightArc, label))
        assert len(possible_transitions) > 0
        return possible_transitions

    def update(self, correct_features, predicted_features):
        keys = set(correct_features) | set(predicted_features)
        for key in keys:
            c = correct_features.get(key, 0.0)
            p = predicted_features.get(key, 0.0)
            self.weights[key] += (c - p) * self.learning_rate
            if self.weights[key] == 0.0:
                del self.weights[key]

    def learn(self, correct_transition, stack, buff, labels, previous_transitions, arcs, input_sentence):
        correct_features = None
        best_features = None
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs, input_sentence)
            score = dot(features, self.weights)
            if best_score == None or score > best_score:
                best_score = score
                best_transition = transition
                best_features = features
            if transition == correct_transition:
                correct_features = features

        if best_transition != correct_transition:
            assert best_features != None
            assert correct_features != None
            self.update(correct_features, best_features)

    def predict(self, stack, buff, labels, previous_transitions, arcs, input_sentence):
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs, input_sentence)
            score = dot(features, self.weights)
            if best_score == None or score > best_score:
                best_score = score
                best_transition = transition
        return (best_score, best_transition)