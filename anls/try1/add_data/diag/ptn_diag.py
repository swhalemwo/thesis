import numpy as np
import os

smpl_props = np.arange(0.2, 0.4, step=0.05)
sim_cutofs = np.arange(0.015, 0.031, step=0.005)
ptn_cnts = [3,4,5]

configs = []

for p in ptn_cnts:
    for c in sim_cutofs:
        for s in smpl_props:
            config = {'pnt_cnt':p, 'sim_cutof':round(c,3), 'smpl_prop':round(s,3)}
            configs.append(config)

basedir = "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/"

# i = configs[30]

for i in configs:
    smpl_prop = i['smpl_prop']
    sim_cutof = i['sim_cutof']
    ptn_cnt = i['pnt_cnt']

    # NEED TO SET ITERATION COUNTER IN BOTH SCRIPTS

    # need to check if graph file already exists -> can save the graph creation for variation in partitions
    # graph are now reused for different number of partitions
    # idk doesn't seem too bad, will not affect the config within  


    flnm = "diag_onmd_smpl_" + str(smpl_prop) + "_sim_"+   str(sim_cutof) + '_it_1_.gt'

    if flnm not in os.listdir(basedir+'graphs/'):
        prep_str = "python3.6 " + basedir + "ptn_diag_prep.py " + str(smpl_prop) + " " + str(sim_cutof)
        os.system(prep_str)

    ptn_str = "python3.6 " + basedir + "ptn_diag_ptn.py " + str(smpl_prop) + " " + str(sim_cutof) + " " + str(ptn_cnt)

    os.system(ptn_str)

    # start at 13:32:50    


