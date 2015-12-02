class Helper:
    INFINITY = 100000000
    LEX_FEAT = 1
    POS_FEAT = 2
    DEP_FEAT = 3

    STACK_SOURCE = 1
    BUFF_SOURCE = 2
    INPUT_SOURCE = 3



    ####################################################################################################################
    ##################################### H E L P E R    F U N C T I O N S #############################################
    ####################################################################################################################



    @staticmethod
    def is_root(word):
        if Helper.get_id(word) == 0:
            return(True)
        else:
            return(False)


    @staticmethod
    def get_id(word):
        try:
            return(int(word[0].strip(' \t\r\n')))
        except:
            return(None)

    @staticmethod
    def get_word(word):
        return(word[1].strip(' \t\r\n'))

    @staticmethod
    def get_stem(word):
        return(word[2].strip(' \t\r\n'))

    @staticmethod
    def get_cpostag(word):
        return(word[3].strip(' \t\r\n'))

    @staticmethod
    def get_postag(word):
        return(word[4].strip(' \t\r\n'))

    @staticmethod
    def get_feats(word):
        return(word[5].strip(' \t\r\n'))

    @staticmethod
    def get_head(word):
        try:
            return(int(word[6].strip(' \t\r\n')))
        except:
            return(None)

    @staticmethod
    def get_deprel(word):
        return(word[7].strip(' \t\r\n'))

    @staticmethod
    def get_phead(word):
        return(int(word[8].strip(' \t\r\n')))

    @staticmethod
    def get_pdeprel(word):
        return(word[9].strip(' \t\r\n'))


    @staticmethod
    def get_leftmost_children(sentence):
        leftmost_children = {}
        for word in sentence:
            lc_candidate = min(leftmost_children.get(Helper.get_head(word), Helper.INFINITY), Helper.get_id(word))
            if lc_candidate < Helper.get_head(word):
                leftmost_children[Helper.get_head(word)] = lc_candidate
        return leftmost_children

    @staticmethod
    def get_rightmost_children(sentence):
        rightmost_children = {}
        for word in sentence:
            rc_candidate = max(rightmost_children.get(Helper.get_head(word), -1), Helper.get_id(word))
            if rc_candidate > Helper.get_head(word):
                rightmost_children[Helper.get_head(word)] = rc_candidate
        return(rightmost_children)

