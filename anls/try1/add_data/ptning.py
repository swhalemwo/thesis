
from graph_tool.all import *
from graph_tool import *
import time
from collections import Counter


g_one_mode = Graph()
g_one_mode.load('one_mode.gt')

tx1 = time.time()
state = minimize_blockmodel_dl(g_one_mode, B_min = 4, B_max = 4)
# state = minimize_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))
tx2 = time.time()

blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_usrs_1md.vertices()]
print(Counter(blks_vlus))

# write state to usrs1k
client.execute('ALTER TABLE usrs1k ADD COLUMN ptn Int8')
client.execute('ALTER TABLE usrs1k UPDATE ptn = 99 where ptn = 0')


for i in g_usrs_1md.vertices():
    idx = g_usrs_1md.vp.id[i]
    ptn = blks[i]

    mod_str = 'ALTER TABLE usrs1k UPDATE ptn = ' + str(ptn) + " where abbrv2 = '" + idx + "'"
    print(mod_str)
    
    client.execute(mod_str)
    



# hm strong increase in memory needed at the end
