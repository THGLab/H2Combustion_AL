from md.md_ts_traj import run_md_simulation
import timeit
"""
The file is a short demo that runs a 10fs molecular dynamics simulation using the active learning model.
It takes in a starting structure (Transition state structure of rxn10), run the simulation with the 
default active learning model(models/model_al_1kperrxn_1) and store outputs in demo/output.
"""
start = timeit.default_timer()

data_dir = "../../md/irc_geom"
rxn = '10'
start_structure = f"{data_dir}/TS_{rxn}.xyz"
output_dir = 'demo_output'
run_md_simulation(start_structure,output_dir,temp = 300, time_fs= 10, stepsize=0.1, calc_option="ML_QBC")

end = timeit.default_timer()
print(f"Time: {end-start} s")


