import pynini as fst 
import pywrapfst
from TL_pckgs.UTILS.tl_sim import TL_sim
from TL_pckgs.UTILS.fst_monitor import FSTMonitor
from TL_pckgs.UTILS.TL_funcs import TL_output_syms_table, periodic_attacker

symbol_table, keys = TL_output_syms_table()
# period = 4
# attack_fst = periodic_attacker(output_syms_table=symbol_table, 
                            #    output_syms_key=keys, period=period)

channel = 'sensor'
# monitor = FSTMonitor(attack_fst)
sensor_attack_period = 16
TLsim = TL_sim(sensor_attack_period=sensor_attack_period, 
               simplified_attack_bool=True )

# path = monitor.build_random_path( max_steps=100 )

# fst_paths_key, fst_paths_letter = monitor.dfs_paths(max_length=4)

# _, path_stateless = monitor.get_path_symbols()

# TLsim.store_attack_sequence(path_stateless, channel=channel)

TLsim.get_unique_attacks(channel=channel, max_length=110, store_bool=True)

rank = int(20)
num_hank_rows = int(30)
num_hank_columns = int(30)

TLsim.learn_attacker(channel=channel, num_hank_columns=num_hank_columns, 
                         num_hank_rows=num_hank_rows, rank=rank, 
                         diagram_boolean=True)

# sup, _, _, bool = TLsim.construct_supervisor()
# print(f"{bool}")