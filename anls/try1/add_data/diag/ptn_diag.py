import numpy as np
import os

import matplotlib.pyplot as plt


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


# * eval
mat_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/mats/'

ptn_cnt = 5
smpl_prop = 0.35
sim_cutof = 0.025

mat_flnm = "ptn_" + str(ptn_cnt) + "_prop_" + str(smpl_prop) + "_ctf_" + str(sim_cutof)

x_mat = np.genfromtxt(mat_dir + mat_flnm + "_clst.csv")
plt.matshow(x_mat)
plt.show()



oks

ptn_cnt = 4
smpl_prop = 0.2
sim_cutof = 0.015
???

ptn_cnt = 3
smpl_prop = 0.2
sim_cutof = 0.015

ptn_cnt = 3
smpl_prop = 0.25
sim_cutof = 0.015, 0.02

ptn_cnt = 3
smpl_prop = 0.3
sim_cutof = 0.015

kinda
ptn_cnt = 3
smpl_prop = 0.3
sim_cutof = 0.025

ptn_cnt = 5
smpl_prop = 0.25
sim_cutof = 0.015

ptn_cnt = 5
smpl_prop = 0.35
sim_cutof = 0.02

ptn_cnt = 5
smpl_prop = 0.35
sim_cutof = 0.025

import csv
import pandas as pd

with open ('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/diag_res.csv', 'r') as fi:
    rd = csv.reader(fi)
    eval_rows = [[float(i) for i in r[0:9]] for r in rd]

eval_df = pd.DataFrame(eval_rows, columns =['ptn_cnt', 'smpl_prop', 'sim_cutof', 'ovrl_sim', 'ptn_sz_sd', 'nbr_vs', 'nbr_es', 'avg_t', 'ptn_sz_sd2'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(eval_df)

eval_df[['ptn_cnt', 'ovrl_sim', 'ptn_sz_sd', 'ptn_sz_sd2']].groupby('ptn_cnt').mean()
# 3 partitions looks best

eval_df[['sim_cutof', 'ovrl_sim', 'ptn_sz_sd', 'ptn_sz_sd2']].groupby('sim_cutof').mean()
# 0.015 looks best

eval_df[['smpl_prop', 'ovrl_sim', 'ptn_sz_sd', 'ptn_sz_sd2']].groupby('smpl_prop').mean()
# unclear 0.3 looks good enough: best in ptn_sz_sd2, almost best in ovrl_sim,
# but ptn_sz_sd most important measure, and 02 does best (0.63, 0.3 has 0.8)

res_vis = np.zeros((4,4))

tgt_vrbl = 'ptn_sz_sd'

c1 = 0
for s in smpl_props:
    c2 = 0
    for c in sim_cutofs:
        try:
            res = eval_df[(eval_df.smpl_prop == s) & (eval_df.sim_cutof == round(c,3)) & (eval_df.ptn_cnt == 5)][tgt_vrbl]
            res_vis[c1,c2] = res
        except:
            res = 0.5
            res_vis[c1,c2] = res
            
        # print(res_vis)

        c2 +=1
    c1+=1


plt.matshow(res_vis)
plt.show()

# vrbl: ovrl sim: low sim_cutofs always preferable
# smpl_prop: increasement good to 0.3, only in 1/4 cases gets better afterwards

# 4 ptns: increase in prop_smp always good
# 5 ptns: increa in prop sample always good, 0.02 best sim_cutof

# ptn_sz_sd:
# 3 ptn: problems only in lowest sample (0.2 and highest cutoff (0.3)
# 4 ptn: lowest sampling best; both lowest and highest cutoff best
# 5 ptn: smpl_prop doesn't matter?, medium cutoffs best (0.02, 0.0.025)

# ptn_sz_sd2
# 3 ptn: 0.3, 0.15 best, unclear general patterns; look good in low (0.015) and medium (0.025) sim cutofs
# 4 ptn: low sim_cutofs best, there high sampling best
# 5 ptn: high sampling, low cutoffs best


# ptn_sz_sd has priority, and it's by far best with 3 partitions
# 3 partitions also best with ovrl_sim
