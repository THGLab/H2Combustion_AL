from md.md_ts_traj import run_md_simulation


data_dir = "../../md/irc_geom"
rxn = '10'
start_structure = f"{data_dir}/TS_{rxn}.xyz"
output_dir = 'demo_output'
run_md_simulation(start_structure,output_dir,temp = 300, time_fs= 10, stepsize=0.1, calc_option="ML_QBC")



