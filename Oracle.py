import sys
from Transition import Transition
from helper_functions import Helper as p

class Oracle:

    def is_removable(self, word, arcs, leftmost_child, rightmost_child):
        if ( (leftmost_child != p.INFINITY and not arcs.get(leftmost_child, False)) or (rightmost_child != -1 and not arcs.get(rightmost_child, False)) ):#CHeck if thie value is correct
            return(False)
        if ( p.is_root(word) ):
            return(False)
        return True

    def getTransition(self, stack, buff, leftmostChildren, rightmostChildren, arcs, labeled):
        """This function should return a Transition object representing the correct action to
        to take according to the oracle."""
        if len(stack) > 1:
            top = stack[-1]
            pre_top = stack[-2]
            rmc_top = rightmostChildren.get(p.get_id(top), -1)
            rmc_pre_top = rightmostChildren.get(p.get_id(pre_top), -1)
            lmc_top = leftmostChildren.get(p.get_id(top), p.INFINITY)
            lmc_pre_top = leftmostChildren.get(p.get_id(pre_top), p.INFINITY)
            if ( p.get_head(pre_top) == p.get_id(top) and self.is_removable(pre_top, arcs, lmc_pre_top, rmc_pre_top) ):
                if labeled:
                    return(Transition(Transition.LeftArc, p.get_deprel(pre_top)))
                else:
                    return(Transition(Transition.LeftArc, None))
            elif ( p.get_head(top) == p.get_id(pre_top) and self.is_removable(top, arcs, lmc_top, rmc_top) ):
                if labeled:
                    return(Transition(Transition.RightArc, p.get_deprel(top)))
                else:
                    return(Transition(Transition.RightArc, None))
            else:
                return(Transition(Transition.Shift, None))
        else:
            if len(buff) >= 1:
                return(Transition(Transition.Shift, None))
            else:
                return(None)


        #assert False, 'Please implement this function!'