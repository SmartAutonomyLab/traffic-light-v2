from __future__ import annotations
import numpy as np 
import pynini as fst  
import time 
import os 
from datetime import datetime 
import pickle 
import random 
import subprocess
from pynini import Fst
from TL_pckgs.UTILS.SCL_funcs import supervisor
from TL_pckgs.UTILS.spec_estimator import SPEC_Estimator
from TL_pckgs.UTILS.TL_funcs import TL_desired_lang, periodic_attacker, TL_input_syms_table, TL_output_syms_table, fst2auto
from TL_pckgs.AUTOMATA.auto_funcs import spwfa2WFA, NTT2FST 
from TL_pckgs.UTILS.fst_monitor import FSTMonitor
from TL_pckgs.WA_package.weighted_automaton import WeightedAutomaton
from typing import Optional, List, Tuple, TYPE_CHECKING    

class TL_sim:
    def __init__(self, sensor_attacker: Optional[fst.Fst] = None, 
                 actuator_attacker: Optional[fst.Fst] = None, 
                 simplified_attack_bool: bool = False,
                 sensor_attack_period: int = 3,
                 actuator_attack_period: int = 3):
        
        # use simplified attack models
        self.simplified_attack_bool = simplified_attack_bool

        # initialize attacker FSTs
        self.sensor_attacker = sensor_attacker
        self.actuator_attacker = actuator_attacker

        # initialize learned attacker FSTs to None
        self.learned_sensor_attacker = None  
        self.learned_actuator_attacker = None

        # initialize attack periods if needed
        self.sensor_attack_period = sensor_attack_period
        self.actuator_attack_period = actuator_attack_period

        # initialize desired language and symbol tables
        self.init_desired_language()

        # initialize attackers if not provided
        self.init_attackers()

        # add monitor for the sensor and actuator attackers 
        # to track attacks and sequences
        self.sensor_attack_monitor = FSTMonitor(self.sensor_attacker)
        self.actuator_attack_monitor = FSTMonitor(self.actuator_attacker)

        # unique sensor and actuator attacks encountered
        # corresponds to FST alphabet where each attack is a 
        # Tuple (input symbol, output symbol). 

        # Also defines the labeling in splearnarray object 
        # for spectral learning postprocessing to construct FSTs
        self.unique_sensor_attacks: List[Tuple] = []
        self.unique_actuator_attacks: List[Tuple] = []

        # current attack sequences being recorded
        # [1, 4, 5, 2, ... ] where each integer corresponds to the 
        # index in the unique attacks list
        self.current_sensor_attack_sequence: List = []  
        self.current_actuator_attack_sequence: List = []

        # list of attack sequences encountered
        # [[1, 2, 5, 6], [3, 4, 2], ... ] where each inner list is a sequence of attacks
        # and each integer corresponds to the index in the unique attacks list
        self.sensor_attacks_sequences: List[List] = []
        self.actuator_attacks_sequences: List[List] = []

    def init_desired_language(self):
        '''
        Initializes the desired language FST for the traffic light system
        and the subsequent input and ouput symbol tables
        '''
        # desired language FST
        self.desired_language = TL_desired_lang()

        # input symbol tables
        input_table, input_keys = TL_input_syms_table()
        self.input_symbol_table = input_table
        self.input_keys = input_keys

        # output symbol tables
        output_table, output_keys = TL_output_syms_table() 
        self.output_symbol_table = output_table
        self.output_keys = output_keys

    def init_attackers(self):    
        '''
        Initializes the sensor and actuator attackers if they are not provided 
        with periodic attack FSTs.
        '''

        if self.sensor_attacker is None:
            # no sensor attacker provided, create periodic sensor attacker
            self.sensor_attacker = periodic_attacker( output_syms_key=self.output_keys, 
                                                     output_syms_table=self.output_symbol_table,
                                                     period=self.sensor_attack_period, 
                                                     simplified_bool=self.simplified_attack_bool)

        if self.actuator_attacker is None:
            # no actuator attacker provided, create periodic actuator attacker
            self.actuator_attacker = periodic_attacker( output_syms_key=self.input_keys, 
                                                       output_syms_table=self.input_symbol_table,
                                                       period=self.actuator_attack_period, 
                                                       simplified_bool=self.simplified_attack_bool)

    def store_attack(self, attack: Tuple[str, str], channel: str, 
                     reset_attack_seq_indicator: Optional[bool] = False):
        '''
        Function to store the attack Tuple
        INPUTS
        attack - Tuple (input symbol, output symbol)
        channel - 'sensor' or 'actuator' to specify which channel the attack occurred on
        reset_attack_seq_indicator - boolean to indicate if the attack sequence should be reset 
                                    after storing the current attack
        '''

        # store unique attacks based on channel
        if channel == 'sensor':
            
            # check if unique sensor attack is recorded
            if attack not in self.unique_sensor_attacks:
                # store and print unique sensor attack
                self.unique_sensor_attacks.append(attack)
                print(f"New sensor attack recorded: {attack}")

            # add attack to current sequence with integer label 
            # corresponding to index in unique attacks list
            attack_index = self.unique_sensor_attacks.index(attack)
            self.current_sensor_attack_sequence.append(attack_index)

            # reset attack sequence if indicated
            if reset_attack_seq_indicator:
                # add current sequence to list
                self.sensor_attacks_sequences.append(self.current_sensor_attack_sequence)
                # reset current sequence
                self.current_sensor_attack_sequence = []

        elif channel == 'actuator':

            # check if actuator attack is unique
            if attack not in self.unique_actuator_attacks:
                # store and print unique actuator attack
                self.unique_actuator_attacks.append(attack)
                print(f"New actuator attack recorded: {attack}")
            
            # add attack to current sequence with integer label 
            # corresponding to index in unique attacks list
            attack_index = self.unique_actuator_attacks.index(attack)
            self.current_actuator_attack_sequence.append(attack_index)

            # reset attack sequence if indicated
            if reset_attack_seq_indicator:
                # add current sequence to list
                self.actuator_attacks_sequences.append(self.current_actuator_attack_sequence)
                # reset current sequence
                self.current_actuator_attack_sequence = []

        else:
            raise ValueError("Channel must be either 'sensor' or 'actuator'")

    def store_attack_sequence(self, attack_sequence: List[Tuple[str, str]], channel: str):
        '''
        Function to store a sequence of attacks
        INPUTS
        attack_sequence - List of attack Tuples (input symbol, output symbol)
        channel - 'sensor' or 'actuator' to specify which channel the attack occurred on
        '''
        if channel == 'sensor' or channel == 'actuator':
            # properly store each attack in the sequence using store_attack function 
            # to track atacks and their unique labels
            for attack in attack_sequence:
                self.store_attack(attack=attack, channel=channel)
            # finalize storage of the attack sequence
            self.finish_storage() 
        else:
            raise ValueError("Channel must be either 'sensor' or 'actuator'")

    def finish_storage(self):
        '''
        Finalizes the storage of attack sequences by appending any ongoing sequences.
        '''
        # finalize sensor attack sequence
        if self.current_sensor_attack_sequence:
            self.sensor_attacks_sequences.append(self.current_sensor_attack_sequence)
            self.current_sensor_attack_sequence = []

        # finalize actuator attack sequence
        if self.current_actuator_attack_sequence:
            self.actuator_attacks_sequences.append(self.current_actuator_attack_sequence)
            self.current_actuator_attack_sequence = []
    
    def get_unique_attacks(self, channel: str
                           , max_length: int, 
                           store_bool: Optional[bool] = True) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[str, str]]]]:
        '''
        Function to get the unique attacks for a given channel from fst attack models
        Args:
        channel (str)- 'sensor' or 'actuator' to specify
          which channel's attacker to observe for unique attacks

        max_length (int) - maximum length of attack sequences to consider for extracting unique attacks
        
        store_bool (bool) - boolean to indicate whether to store the unique attacks in the class attribute for the channel;
        
        Returns:
        fst_path_keys (List[List[Tuple[int, int]]]) - list of unique attack sequences keys 
        observed for the specified channel

        fst_path_keys (List[List[Tuple[str, str]]]) - list of unique attack sequences labels 
        observed for the specified channel
        '''

        # get monitor for the specified channel
        if channel == 'sensor':
            monitor = self.sensor_attack_monitor
        elif channel == 'actuator':
            monitor = self.actuator_attack_monitor
        else:
            raise ValueError("Channel must be either 'sensor' or 'actuator'")
        
        # get paths through the attacker FST up to max_length and extract unique attacks
        fst_paths_keys, fst_paths_letter = monitor.dfs_paths(max_length=max_length)

        if store_bool:
            # store letters for unique attacks in class attribute for the channel
            # since the pair will be given unique integer labels needed for spectral learning
            for path in fst_paths_letter:
                self.store_attack_sequence(attack_sequence=path, channel=channel)

        return fst_paths_keys, fst_paths_letter

    def learn_attacker(self, channel: str, 
                           rank: Optional[int] = None, 
                           num_hank_rows: Optional[int] = None, 
                           num_hank_columns: Optional[int] = None, 
                           diagram_boolean: Optional[bool] = False, 
                           save_learned_model: bool = True,
                           diagram_file_name: str = "attacker_fst.pdf") -> fst.Fst:
        '''
        Constructs an attacker FST based on the unique attacks recorded
        for the specified channel.
        INPUTS
        channel - 'sensor' or 'actuator' to specify which channel's attacker to construct
        rank - rank parameter for hankel matrices spectral learning
        num_hank_rows - number of rows in hankel basis for spectral learning
        num_hank_columns - number of columns in hankel basis for spectral learning
        diagram_boolean - boolean to indicate if a diagram of the learned attacker should be saved
        diagram_file_name - filename to save the diagram if diagram_boolean is True
        OUTPUTS
        attacker_fst - constructed attacker FST (sensor or actuator)
        '''

    

        # choose unique attacks and attack sequences based on channel for construction
        if channel == 'sensor':
            unique_attacks = self.unique_sensor_attacks
            attack_sequences = self.sensor_attacks_sequences

            # input and output symbols should e the same
            symbol_table = self.sensor_attacker.input_symbols()
        elif channel == 'actuator':
           unique_attacks = self.unique_actuator_attacks
           attack_sequences = self.actuator_attacks_sequences

           # input and output symbols should be the same
           symbol_table = self.actuator_attacker.input_symbols()
        else:
            raise ValueError("Channel must be either 'sensor' or 'actuator'")

        # set hankel parameters if not provided for spectral learning
        if rank is None:
            rank = len(unique_attacks)
        if num_hank_rows is None:
            num_hank_rows = 2 * len(unique_attacks)
        if num_hank_columns is None:
            num_hank_columns = 2 * len(unique_attacks)

        # initialize spectral estimator
        SPEC_ESTIMATOR = SPEC_Estimator( num_hank_columns=num_hank_columns,
                                  num_hank_rows=num_hank_rows,
                                  rank=rank,
                                  num_letters=len(unique_attacks) )
        
        # build splearnarray object from list of attack sequences
        SPEC_ESTIMATOR.construct_splearnarray(words_list_num=attack_sequences)

        # begin to track time
        start_time = time.perf_counter()

        # perform spectral analysis
        SPEC_ESTIMATOR.build_spec_est()

        NTT = spwfa2WFA( SPEC_ESTIMATOR.spec_EST.NTT )

        attacker_fst = NTT2FST( NTT=NTT,
                                input_table=symbol_table, 
                                output_table=symbol_table,
                                pair_list=unique_attacks
                                )
        
        # spectral methods have finished
        finish_time = time.perf_counter()

        # sort arcs from each state
        attacker_fst.arcsort()

        # remove epsilon transitions
        attacker_fst.rmepsilon()
        
        # map to automaton version
        attacker_auto = fst2auto(f=attacker_fst, 
                                 input_table=self.input_symbol_table, 
                                 output_table=self.output_symbol_table)
        # make deterministic
        attacker_auto = fst.determinize(attacker_auto)

        # make daigram if commanded
        if diagram_boolean:
            # 1) Write DOT
            attacker_fst.draw("attacker_fst.dot")

            # 2) Render to PDF (requires Graphviz 'dot' on PATH)
            subprocess.run(["dot", "-Tpdf", "attacker_fst.dot", "-o", diagram_file_name], check=True)

            print(f"Diagram of attacker saved to {diagram_file_name}")
        
        # save learned model data
        if save_learned_model:
            self.save_learned_fst_data(start_time=start_time, 
                                       finish_time=finish_time,
                                       attack_fst=attacker_fst, 
                                       attack_sequences=attack_sequences)

        # store learned attacker FST in the appropriate attribute
        if channel == 'sensor':
            self.learned_sensor_attacker = attacker_fst
        elif channel == 'actuator':
            self.learned_actuator_attacker = attacker_fst
                                
        return attacker_fst
    
    def save_learned_fst_data(self, start_time: float, finish_time: float, 
                              attack_fst: fst.Fst, 
                              attack_sequences: list[list[int]]):
        '''
        Saves run time, number of attack sequences needed to learn model and attack_fst model
        
        :param self: 
        :param start_time: internal machine time for when spectral learning algorithm began
        :type start_time: float
        :param finish_time: internal machine time for when spectral learning algorithm began
        :type finish_time: float
        :param attack_fst: fst model for learned atacker model
        :type attack_fst: fst.Fst
        :param attack_sequences: list of attack sequences [[ 1, 2, 4 ...], [1, 5, 9], ...]
        :type attack_sequences: list[list[int]]
        '''

        # Count how many attack sequences were used to obtain attack_fst learned model
        # initialize total number of sequences to zero
        num_sequences = 0.0
        for attack_list in attack_sequences:
            num_sequences += len(attack_list)

        # calculate run time of spectral learning algorithm
        run_time = finish_time - start_time

        # Package data
        data = {
            "run_time_seconds": run_time,
            "num_sequences": num_sequences,
            "attack_fst": attack_fst
        }

        print(f"run time is {run_time} and sequence total is {num_sequences}")

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learned_model_info_{timestamp}.pkl"

        # Ensure output directory exists
        output_dir = "learned_models"
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        # Save pickle
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved learned model data to {filepath}")

    def construct_supervisor(self, As: Optional[fst.Fst] = None, Aa: Optional[fst.Fst] = None) -> Tuple[fst.Fst, fst.Fst, fst.Fst, bool]:
        '''
        Construct FST supervisor using attacker FSTs and desired language 
        with supervisory control theory synthesis.
        
        Args:
        Aa (fst.Fst) - actuator attacker
        As (fst.Fst) - sensor attacker

        Returns:
        supervisor_fst - the synthesized supervisor FST
        MK_auto - the automaton for the desired language
        control_auto - the automaton for the corrupted supervisor's language (the allowed language under attack
        controllable - boolean indicating if the desired language is controllable under the given attackers
        '''

        if Aa is None:
            # use learned actuator attacker model if none is provided
            Aa=self.learned_actuator_attacker

        if As is None:
            # use learned sensor attacker if none is provided
            As=self.learned_sensor_attacker

        supervisor_fst, MK_auto, control_auto, controllable = supervisor( MK=self.desired_language, 
                                                                         Aa=Aa, 
                                                                         As=As)

        return supervisor_fst, MK_auto, control_auto, controllable
    
    