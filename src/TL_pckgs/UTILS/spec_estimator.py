from datetime import datetime
import os
import pickle 
import numpy as np
import random 
import sympy as sp
from splearn.datasets.data_sample import SplearnArray
from splearn.datasets.base import DataSample
from splearn.spectral import Spectral
from typing import Optional 

class SPEC_Estimator:
    def __init__(self, 
                 num_letters: int,
                 rank: Optional[int] = None, 
                 num_hank_rows: Optional[int] = None,
                 num_hank_columns: Optional[int] = None,
                 show_WFA: Optional[bool] = False,
                 NTT_ind: Optional[bool] = True):
        """
        spectral estimator for learning WFA and a transition tuple from 
        observed words

        INPUTS
        num_hank_rows - the m in m x n hankel matrix created by spectral.fit method
        num_hank_columns - the n in m x n hankel matrix created by spectral.fit method
        show_WFA - prints WFA learned from data if true, set to false on default
        USEFUL ATTRIBUTES
        est - instance of estimator class with WFA attached from scikit-splearn
        inp - data input for est.fit() method that is unscaled 
                DO NOT UNDERSTAND STRUCTURE
                it housees num_samples, num_letters and the strings themselves
        inp_scaled - data input for est.fit() method 
                that is scaled by the weights calculated with BT inferencing 
                DO NOT UNDERSTAND STRUCTURE
                it housees num_samples, num_letters and the strings themselves
        """
        # estimator parameters

        # number of letters in the alphabet
        self.num_letters = num_letters

        # Approximate rank of hankel matrices that will be constructed
        self.rank = rank

        # approximate size of hankel basis (rows and columns)
        self.num_hank_rows = num_hank_rows
        self.num_hank_columns = num_hank_columns

        # number of words in the sample (post data processing)
        self.num_words = None

        # indicator to show learned WFA
        self.show_WFA = show_WFA

        # indicator to comptue relevant NTT
        self.NTT_ind = NTT_ind

        # special arrays used for input into splearn estimator
        self.splearn_array_unique_words = None

        # estimator from scaled data
        self.spec_EST = None 

    def construct_splearnarray(self, words_list_num):
        """
        INPUTS
        words_list - [list1, list2, ..., listn] where listi = [1 5 0 8 5 4] each list should be unique
        OUTPUTS
        **SplearnArray** class inherit from numpy ndarray as a 2d data ndarray.
        
        Example of a possible 2d shape:
        
        +---+---+---+---+---+
        |  0|  1|  0|  3| -1|
        +---+---+---+---+---+
        |  0|  0|  3|  3|  1|
        +---+---+---+---+---+
        |  1|  1| -1| -1| -1|
        +---+---+---+---+---+
        |  5| -1| -1| -1| -1|
        +---+---+---+---+---+
        | -1| -1| -1| -1| -1|
        +---+---+---+---+---+
        
        is equivalent to:
        
        - word (0103) or abad
        - word (00331) or aaddb
        - word (11) or bb
        - word (5) or f
        - word () or empty
        
        Each line represents a word of the sample. The words are represented by integer letters (0->a, 1->b, 2->c ...).
        -1 indicates the end of the word. The number of rows is the total number of words in the sample (=nbEx) and the number of columns
        is given by the size of the longest word. Notice that the total number of words does not care about the words' duplications. 
        If a word is duplicated in the sample, it is counted twice as two different examples. 
        
        The DataSample class encapsulates also the sample's parameters 'nbL', 'nbEx' (number of letters in the alphabet and 
        number of samples) and the fourth dictionaries 'sample', 'prefix', 'suffix' and 'factor' that will be populated during the fit
        estimations.
        """
        self.words_list_num = words_list_num

        length_list = [len(word) for word in words_list_num]

        max_word_length = max(length_list)

        for index, word in enumerate( words_list_num ):

            # info about current sequence of events
            len_word   = length_list[index]
            word_array = np.array(word)

            # map word into splearn format
            word_array_extended = -1*np.ones( (1,max_word_length) )
            word_array_extended[0, :len_word] = word_array

            complete_prefix_array = self.repeat_construct(prefix_word_array=word_array_extended, 
                                                          len_prefix=len_word)
            if index == 0:
                # first word, initialize splearn array
                array_words = complete_prefix_array
            else:
                # stack new word to the array
                array_words = np.vstack( (array_words, complete_prefix_array) )

        
        self.num_words = array_words.shape[0]
        self.splearn_array = SplearnArray(array_words)
        # self.num_words = 1
        # self.splearn_array = SplearnArray(array_words[0,:].reshape(1,-1))

    def repeat_construct(self,prefix_word_array: list[int], 
                         len_prefix: int):
        """
        Constructs a 2d array of prefixes of the word in prefix_word_array
        INPUTS
        prefix_word_array - list of integers representing a word [4, 0, 1, 2, 3]
        len_prefix - length of the word in prefix_word_array
        OUTPUTS
        prefix_array_complete - 2d array of prefixes of the word in prefix_word_array
        """
        for ind in range(len_prefix,0,-1):

            # initialize prefix, sequence of events decreases in length with each iteration
            prefix_word_array[0, ind:] = - 1
            
            if ind == len_prefix:
                # no stacking necessary for first iteration
                prefix_array_complete = prefix_word_array.copy()
            else:
                # stack new prefix to the array
                prefix_array_complete = np.vstack( 
                                                (prefix_array_complete, 
                                                prefix_word_array) 
                                                )
        return prefix_array_complete

    def build_spec_est(self):

        # construct tuple for splearn data sample 
        data_tuple  = (self.num_letters, self.num_words, self.splearn_array)

        # map tuples to data sample form for scikit splearn package
        data_sample = DataSample(data_tuple)

        # obtain data object from data samples 
        inp_unique = data_sample.data 

        # initialize spectral estimator
        self.spec_EST = Spectral()
        self.spec_EST.set_params(lrows=self.num_hank_rows, lcolumns=self.num_hank_columns, 
                    smooth_method=None , rank=self.rank,
                        version="classic",partial=True )
        
        # apply spectral learning to scaled data 
        self.spec_EST.fit(inp_unique, NTT_ind=self.NTT_ind, full_svd=True)
        
        self.hankels  = self.spec_EST._hankel
        learned_NTT   = self.spec_EST.NTT
        # splearn_WFA = est.automaton
        # self.learned_WFA = spwfa2WFA(splearn_WFA, alphabet=self.WFA.alphabet)

        if self.NTT_ind:
            print(f"initial: {self.spec_EST.automaton.initial}")
            print(f"final: {self.spec_EST.automaton.final}")
            print(f"transitions: {self.spec_EST.automaton.transitions}")

            print(f"NTT initial: {learned_NTT.initial}")
            print(f"NTT final: {learned_NTT.final}")
            print(f"NTT transitions: {learned_NTT.transitions}") 

def scores2samples(scores):
    """
    INPUTS
    scores - list of scores from BT inference
    
    OUTPUTS
    scaled - 
    """
    count = len(scores)
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True) 

    return indexed_scores

def save_bts_est(bts_est, folder_path):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"bts_est_{timestamp}.pkl"
    full_path = os.path.join(folder_path, filename)

    # Save the object
    with open(full_path, "wb") as f:
        pickle.dump(bts_est, f)

    print(f"Saved BTS Estimator to: {full_path}")
    return full_path  # Return the path so it can be logged or reused

def load_bts_est(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        bts_est = pickle.load(f)
    print(f"Loaded BTS Estimator from: {pickle_path}")
    return bts_est

