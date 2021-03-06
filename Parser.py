import sys
import time
from Oracle import Oracle
from Perceptron import PerceptronModel
from Transition import Transition
from helper_functions import Helper as h
from SVM import *



class Parser:

    def __init__(self, labeled):
        self.labeled = labeled

    def initialize(self, sentence):
        self.root = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '-1', 'ROOT', 'ROOT', 'ROOT']
        self.buff = [self.root] + list(reversed(sentence))
        self.stack = list()
        self.arcs = {} #arcs is actually a mapping from child to parent (child's head)
        self.labels = {}
        self.transitions = list()
        self.leftmostChildren = h.get_leftmost_children(sentence) #Map from parent to leftmost child
        self.rightmostChildren = h.get_rightmost_children(sentence) #Map from parent to rightmost child


    def execute_transition(self, transition):
        """This function should take a transition object and apply to the
    	current parser state. It need not return anything."""
        self.transitions.append(transition.transitionType)
        if (transition.transitionType == Transition.Shift):
            self.stack.append(self.buff.pop())
        elif (transition.transitionType == Transition.LeftArc):
            top = self.stack.pop()
            top_id= h.get_id(top)
            pre_top = self.stack.pop()
            pre_top_id = h.get_id(pre_top)
            self.stack.append(top)
            self.arcs[pre_top_id] = top_id
            self.labels[pre_top_id] = transition.label
        else:
            top = self.stack.pop()
            top_id = h.get_id(top)
            pre_top = self.stack[-1]
            pre_top_id = h.get_id(pre_top)
            self.arcs[top_id] = pre_top_id
            self.labels[top_id] = transition.label


    @staticmethod
    def load_corpus(filename):
        print >>sys.stderr, 'Loading treebank from %s' % filename
        corpus = []
        sentence = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    corpus.append(sentence)
                    sentence = []
                else:
                    word = line.split('\t')
                    sentence.append(word)
        print >>sys.stderr, 'Loaded %d sentences' % len(corpus)
        return corpus

    def output(self, sentence):
        for token in sentence:
            head = self.arcs.get(h.get_id(token), '0')
            label = self.labels.get(h.get_id(token), '_')
            label = label if label is not None else '_'
            token[6] = str(head)
            token[7] = str(label)
            print '\t'.join(token)
        print


    def train(self, trainingSet, model):
        corpus = Parser.load_corpus(trainingSet)
        oracle = Oracle()
        for sentence in corpus:
            self.initialize(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                transition = oracle.getTransition(self.stack, self.buff, \
                    self.leftmostChildren, self.rightmostChildren, \
                    self.arcs, self.labeled)
                #model.learn(transition, self.stack, self.buff, \
                #    self.labels, self.transitions, self.arcs, sentence)
                model.compile_svm_feats(transition, self.stack, self.buff, \
                    self.labels, self.transitions, self.arcs, sentence)
                self.execute_transition(transition)
        model.train_svm()


    def parse(self, testSet, model):
        corpus = Parser.load_corpus(testSet)
        for sentence in corpus:
            self.initialize(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                #_, transition = model.predict(self.stack, self.buff, \
                #    self.labels, self.transitions, self.arcs, sentence)
                transition = model.predict_svm(self.stack, self.buff, \
                    self.labels, self.transitions, self.arcs, sentence)
                self.execute_transition(transition)
            self.output(sentence)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', '-l', action='store_true')
    parser.add_argument('trainingcorpus', help='Training treebank')
    parser.add_argument('testset', help='Dev/test treebank')
    args = parser.parse_args()

    p = Parser(args.labeled)


    #perceptron_model = PerceptronModel(args.labeled)

    svm_model = SVMModel(args.labeled)

    p.train(args.trainingcorpus, svm_model)
    p.parse(args.testset, svm_model)
