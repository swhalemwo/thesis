# doesn't need to get usr_trk_lnks all the time, can just save once and read back in

from graph_tool.all import *
from graph_tool import *
import numpy as np
import argparse
import itertools
import csv
import random
import time


def vd_fer(g, idx):
    """creates vertex index dict from graph and id property map"""
    # vertex_dict
    vd = {}
    vdrv = {}
    for i in g.vertices():
        vd[idx[i]] = int(i)
        vdrv[int(i)] = idx[i]
        
    return(vd, vdrv)

print('construct graph')
basedir = "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/"
with open(basedir + 'usr_trk_lnks.csv', 'r') as fi:
    rdr = csv.reader(fi)
    lnks = [(r[0], r[1], int(r[2])) for r in rdr]

unq_usrs = np.unique([i[0] for i in lnks])

g_usrs = Graph()
g_usrs.edge_properties['plcnt'] = g_usrs.new_edge_property('int')

g_usrs.vp['id'] = g_usrs.add_edge_list(lnks, hashed = True, string_vals = True, eprops = [g_usrs.ep.plcnt])
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs.vp.id)

# smpl_ep = g_usrs.new_edge_property('bool')


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('smpl_prop', help='proportion of songs to select of user')
    parser.add_argument('sim_cutof', help='how similar two users have to be to be connected')

    args = parser.parse_args()
    smpl_prop = float(args.smpl_prop)
    sim_cutof = float(args.sim_cutof)
    
    # smpl_prop = 0.3
    # sim_cutof = 0.02

    for it in range(1,9):
        smpl_ep = g_usrs.new_edge_property('bool')

        print('sample graph')
        for u in unq_usrs:

            vu = g_usrs.vertex(g_usrs_vd[u])
            u_dg_org = vu.out_degree()

            alctd_dg = u_dg_org*smpl_prop
            new_deg = 0

            sngs_el = list(vu.out_edges())
            random.shuffle(sngs_el)

            for e in sngs_el:
                new_deg = new_deg + 1
                smpl_ep[e] = True

                if new_deg > alctd_dg:
                    break

        g_usrs2 = Graph(GraphView(g_usrs, efilt = smpl_ep), prune=True)

        g_usrs2_vd, g_usrs2_vd_rv = vd_fer(g_usrs2, g_usrs2.vp.id)

        print('calculate similarities')
        sample_ids = [g_usrs2_vd[i] for i in unq_usrs]

        usr_cmps = list(itertools.combinations(sample_ids, 2))

        t1 = time.time()
        smpl_sims = vertex_similarity(g_usrs2, 'dice', vertex_pairs = usr_cmps, eweight = g_usrs2.ep.plcnt)
        t2 = time.time()

        print('construct 1 mode edgelist')
        usr_lnks = np.where(smpl_sims > sim_cutof)

        elx = []
        for i in usr_lnks[0]:
            rel_e =usr_cmps[i]
            elx.append((g_usrs2_vd_rv[rel_e[0]], g_usrs2_vd_rv[rel_e[1]]))

        g_usrs_1md = Graph(directed=False)
        g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True)

        g_usrs_1md.vertex_properties['id'] = g_usrs_1md_id

        flnm = basedir + "graphs/diag_onmd_smpl_" + str(smpl_prop) + "_sim_"+   str(sim_cutof) + '_it_' + str(it) + '.gt'
        g_usrs_1md.save(flnm)


   

