import random
from typing import List, Tuple, Optional
from TL_pckgs.AUTOMATA.auto_funcs import auto_paths
from TL_pckgs.UTILS.TL_funcs import fst2auto, auto_key2fst_key
import pynini

class FSTMonitor:
    """Monitors a PyINI FST by randomly selecting feasible output symbols and tracking paths."""
    
    def __init__(self, fst: pynini.Fst):
        """
        Initialize the FST monitor.
        
        Args:
            fst: A PyINI FST to monitor
        """
        self.fst = fst
        self.input_table = fst.input_symbols()
        self.output_table = fst.output_symbols()
        self.path_history: List[Tuple[int, int, int, int]] = []
        self.current_state = fst.start()
    
    def get_feasible_outputs(self, state: int, chosen_input_key: Optional[int] = None) -> List[Tuple[int, int, int]]:
        """
        Get feasible output symbols from a given state.
        
        Args:
            state: The current state
            chosen_input_key: The input symbol key to consider (None for random choice)
            
        Returns:
            List of (input_key, output_key, next_state) tuples
        """
        feasible = []
        if chosen_input_key is None:
            # Collect all feasible transitions
            for arc in self.fst.arcs(state):
                input_key = arc.ilabel if arc.ilabel else 0
                output_key = arc.olabel if arc.olabel else 0
                feasible.append((input_key, output_key, arc.nextstate))
        else:
            # Collect transitions matching the chosen input key
            for arc in self.fst.arcs(state):
                if arc.ilabel == chosen_input_key:
                    input_key = arc.ilabel if arc.ilabel else 0
                    output_key = arc.olabel if arc.olabel else 0
                    feasible.append((input_key, output_key, arc.nextstate))
        return feasible
    
    def step(self, chosen_input_key: Optional[int] = None) -> Optional[Tuple[int, int, int]]:
        """
        Take a random step through the FST.
        
        Returns:
            Tuple of (input_key, output_key, next_state) or None if no feasible transitions
        """
        feasible = self.get_feasible_outputs(self.current_state, 
                                             chosen_input_key=chosen_input_key)
        if not feasible:
            return None
        
        input_key, output_key, next_state = random.choice(feasible)
        self.path_history.append((self.current_state, next_state, input_key, output_key))
        self.current_state = next_state
        return (input_key, output_key, next_state)
    
    def reset(self):
        """Reset the monitor to the initial state."""
        self.current_state = self.fst.start()
        self.path_history = []
    
    def get_path_symbols(self) -> Tuple[ List[Tuple[int, int, str, str]], 
                                        List[Tuple[str, str]] ]:
        """
        Get the recorded path history with symbol strings instead of keys.
        Returns:
            path_with_symbols - List of tuples (current_state, next_state, input_symbol, output_symbol)
            stateless_path    - List of tuples (input_symbol, output_symbol) with strings
        """
        path_with_symbols = []
        stateless_path    = []

        for current_state, next_state, input_key, output_key in self.path_history:
            input_sym = self.input_table.find(input_key) if input_key else ""
            output_sym = self.output_table.find(output_key) if output_key else ""
            path_with_symbols.append((current_state, next_state, input_sym, output_sym))
            stateless_path.append((input_sym, output_sym))
        return path_with_symbols, stateless_path
    
    def build_random_path(self, max_steps: int, 
                          chosen_inputs: Optional[list[int]] = None) -> List[Tuple[int, int, int, int]]:
        """
        Build a random path through the FST up to max_steps.
        
        Args:
            max_steps: Maximum number of steps to take
            chosen_inputs: Optional list of input keys to use at each step
            
        Returns:
            The recorded path history
        """
        self.reset()
        
        if chosen_inputs is None:
            for _ in range(max_steps):
                if self.step() is None:
                    break
        else:
            for input_key in chosen_inputs:
                if self.step(chosen_input_key=input_key) is None:
                    break
        return self.path_history
    
    def dfs_paths(self, max_length: int, include_all: Optional[bool] = False) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[str, str]]] ]:
        """
        Get all accepted paths through the FST up to a certain length using depth-first search.
        This method is currently set up to return only paths of length exactly equal to max_length, 
        but can be modified to return all paths up to max_length with include_all boolean.

        Args:
            max_length (int): Maximum length of paths to consider

            include_all (bool): If True, include all paths of length <= max_length; 
            if False, include only paths of length == max_length
        """

        # first convert the FST into an automaton (each transition has same input and output symbol)
        auto = fst2auto(self.fst)

        # then use the auto_paths function to get all accepted paths of the automaton up to max_length
        auto_paths_key, auto_paths_letter, longest_paths_key, longest_paths_letter = auto_paths(auto, max_length)
        if include_all:
            # convert the auto paths keys of all length to FST paths keys format
            fst_paths_keys, fst_paths_letters = self.auto_paths_keys2fst_paths_keys(auto_paths_key=auto_paths_key)

            return fst_paths_keys, fst_paths_letters
        else:
            # convert the auto paths keys of length == max_length to FST paths keys format
            fst_paths_keys, fst_paths_letters = self.auto_paths_keys2fst_paths_keys(auto_paths_key=longest_paths_key)

            return fst_paths_keys, fst_paths_letters
    
    def auto_paths_keys2fst_paths_keys(self, auto_paths_key: List[List[int]]) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[str, str]]], ]:
        """
        Convert auto paths keys to FST paths keys format.
        
        Args:
            auto_paths_key: List of accepted paths in the automaton format 
            (list of lists of concatenated input/output keys)
            
        Returns:
            fst_paths_keys: List of accepted paths in the FST format (list of lists of tuples (input_key, output_key))
            fst_paths_letter: List of accepted paths in the FST format (list of lists of tuples (input_symbol, output_symbol))
        """
        # initialize empty list for FST paths keys and letters
        fst_paths_keys = []
        fst_paths_letters = []
        # iterate through each path in the automaton list of paths or "words"
        for path in auto_paths_key:
            # initialize/reset a list to store the current path in the FST format
            current_path_keys = []
            current_path_letters = []
            # iterate through each concatenated input/output key in the path
            for auto_key in path:

                # convert the auto key to FST input and output keys using the auto_key2fst_key function
                input_key, output_key = auto_key2fst_key(auto_key)

                # get the corresponding input and output symbols from the FST symbol tables
                input_letter = self.input_table.find(input_key) 
                output_letter = self.output_table.find(output_key)

                # add to current path
                current_path_letters.append((input_letter, output_letter))
                current_path_keys.append((input_key, output_key))

            # add finished path to list of FST paths
            fst_paths_keys.append(current_path_keys)
            fst_paths_letters.append(current_path_letters)

        return fst_paths_keys, fst_paths_letters

       
        

