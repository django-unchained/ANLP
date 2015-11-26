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

    def extract_features(self, transition, stack, buff, labels, previous_transitions, arcs):
        features = defaultdict(float)
        tType = transition.transitionType
        label = transition.label

        # Top two POS tags from the stack
        for i in range(2):
            if i >= len(stack):
                break
            s = stack[-(i+1)]
            pos = s[3]
            features['transition=%d,s%d.pos=%s' % (tType, i, pos)] = 1

        # Next four POS tags from the buffer
        for i in range(4):
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
            features['transition=%d,dist=%d' % (tType, dist)] = 1

        #Valency function
        if len(stack) > 1:
            if tType == 1: # Left Arc
                [left_valency, right_valency] = self.get_valency(arcs, h.get_id(stack[-1]))
                features['transition=%d,head_left_valency=%d' % (tType, left_valency)] = 1
                features['transition=%d,head_right_valency=%d' % (tType, right_valency)] = 1
            else:
                [left_valency, right_valency] = self.get_valency(arcs, h.get_id(stack[-2]))
                features['transition=%d,head_left_valency=%d' % (tType, left_valency)] = 1
                features['transition=%d,head_right_valency=%d' % (tType, right_valency)] = 1
        return features

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

    def learn(self, correct_transition, stack, buff, labels, previous_transitions, arcs):
        correct_features = None
        best_features = None
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs)
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

    def predict(self, stack, buff, labels, previous_transitions, arcs):
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs)
            score = dot(features, self.weights)
            if best_score == None or score > best_score:
                best_score = score
                best_transition = transition
        return (best_score, best_transition)