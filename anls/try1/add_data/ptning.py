
from graph_tool.all import *
from graph_tool import *
import time
from collections import Counter


g_one_mode = Graph()
g_one_mode.load('one_mode1k_002.gt')
g_one_mode.load('one_mode1k_004.gt')
g_one_mode.load('one_mode1k_dice_005.gt')
g_one_mode.load('one_mode4k_dice_002_smpl.gt')
g_one_mode.load('one_mode4k_dice_00175_smpl.gt')
g_one_mode.load('one_mode4k_dice_002_smpl_l025.gt')


g_one_mode.load('one_mode4k_004.gt')



# ** section bisection_args
# help(graph_tool.inference.mcmc.mcmc_multilevel)
mcmc_multilevel_args = {'r':10}

bisection_args = {'mcmc_multilevel_args':mcmc_multilevel_args}
# help(graph_tool.inference.bisection.bisection_minimize)
# has further mcmc_multilevel_args
# help(graph_tool.inference.mcmc.mcmc_multilevel)


# help(graph_tool.inference.blockmodel.BlockState.entropy)
entropy_args = {}
entropy_args = {'multigraph':False, 'exact':True}
# 'recs':True, 'recs_dl':True

mcmc_args = {'parallel':True, 'entropy_args':entropy_args, 'd':0.01, 'beta':1, 'niter':1}
# help(graph_tool.inference.blockmodel.BlockState.mcmc_sweep)
# impact of d: no real difference between 0.01 and 0.1
# beta:0.1 slower (2.85)
# beta 10: bit faster (2.45)
# seems to be spurious tho


# help(graph_tool.inference.mcmc.mcmc_equilibrate)
mcmc_equilibrate_args = {'mcmc_args':mcmc_args, 'wait':1, 'max_niter':4}

# not clear what default values are, putting in the seemingly default ones makes it work forever
# wait seems to be default 1
# max_iter: maybe 5? 1 decreases by ~10%
# maybe values are dynamic, and depend on other stuff? 

# help(graph_tool.inference.blockmodel.BlockState.merge_sweep)
shrink_kargs = {'niter':3, 'entropy_args':entropy_args, 'parallel':True}

# help(graph_tool.inference.blockmodel.BlockState.shrink)

shrink_args = {'**kwargs':shrink_kargs}



bisection_args = {}
entropy_args = {}
mcmc_equilibrate_args = {}
shrink_args = {}

mcmc_args = {}

ts = []
for i in range(5):

    t1 = time.time()
    state = minimize_blockmodel_dl(g_one_mode, B_min = 4, B_max = 4,
                                   bisection_args = bisection_args,
                                   mcmc_args  = mcmc_args,
                                   mcmc_equilibrate_args = mcmc_equilibrate_args,
                                   shrink_args = shrink_args,
                                   deg_corr = True)
    t2 = time.time()

    print(t2-t1)
    ts.append(t2-t1)
np.mean(ts)


# effects unclear:
# seeting multigraph, exact to false seems good
# parallel also good
# mcmc_equilibrate_args seems to increase time? 
blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_one_mode.vertices()]
print(Counter(blks_vlus))



statez = graph_tool.inference.minimize.get_states(g_one_mode, B_min = 4, B_max = 4, dense=True)

check entropy args?

xstate=graph_tool.inference.blockmodel_em.EMBlockState(g_one_mode, 4, init_state=None)
state_x2 = EMBlockState(g_one_mode, B=4)

delta, niter = em_infer(state_x2)

# deg_corr = False seems to increase by ~ 10%
# parallel=True increases speed by 25%
# mcmc_args niter = 0 increases speeed a lot, not sure if it complete breaks things tho

# no impact by 
# 'entropy_args':{'multigraph':False},
# sequential = True
# 'c':10,
# setting 'beta' even to default (1) seems to increase time??
# seems to be same with mcmc_equilibrate wait: even calling with default value increases time
# or defaults are wrongly documented


# no settings changed: 24 sec
# mcmc_

# graph_tool.inference.blockmodel.BlockState.mcmc_sweep()
#  mcmc_equilibrate()


blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_one_mode.vertices()]
print(Counter(blks_vlus))

# write state to usrs1k
# client.execute('ALTER TABLE usrs1k ADD COLUMN ptn Int8')
# client.execute('ALTER TABLE usrs1k UPDATE ptn = 99 where ptn = 0')


for i in g_one_mode.vertices():
    idx = g_one_mode.vp.id[i]
    ptn = blks[i]

    mod_str = 'ALTER TABLE usrs4k UPDATE ptn = ' + str(ptn) + " where abbrv2 = '" + idx + "'"
    print(mod_str)
    
    client.execute(mod_str)
    



# hm strong increase in memory needed at the end


# state = minimize_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))
