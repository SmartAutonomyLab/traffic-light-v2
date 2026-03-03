import time
from TL_pckgs.UTILS.tl_sim import TL_sim
from TL_pckgs.UTILS.TL_funcs import periodic_attacker

TLSIM = TL_sim()

# tables for periodic attacker construction
input_symbol_table = TLSIM.input_symbol_table
input_keys = TLSIM.input_keys

output_symbol_table = TLSIM.output_symbol_table
output_keys = TLSIM.output_keys

# attacker sizes
sensor_period_list = [4, 8, 12, 16]
actuator_period_list = [ 4, 8, 12, 16]

for sensor_period in sensor_period_list:
    for actuator_period in actuator_period_list:

        # build attacker FSTs
        sensor_attacker = periodic_attacker(output_syms_table=output_symbol_table, 
                                            output_syms_key=output_keys, 
                                            period=sensor_period)

        actuator_attacker = periodic_attacker(output_syms_key=input_keys, 
                                            output_syms_table=input_symbol_table, 
                                            period=actuator_period)

        start_time = time.perf_counter()
        _, _, _, controllable = TLSIM.construct_supervisor(As=sensor_attacker, 
                                Aa=actuator_attacker)
        finish_time = time.perf_counter()

        # time to construct 
        construct_time = finish_time - start_time

        print(f"controllable bool is {controllable}")
        print(f"Sensor period: {sensor_period}\\Actuator period: {actuator_period}\\Construct time: {construct_time} ")