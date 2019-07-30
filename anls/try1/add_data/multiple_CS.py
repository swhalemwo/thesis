

# ** induce hierarchy for a block
subset dfc?
need new playcount

maybe best to make new temp table and merge with mbid_id

ptn = 0
one_md_usrs = [g_usrs_1md_id[i] for i in g_usrs_1md.vertices()]
ptn_usrs = np.array(one_md_usrs)[np.where(np.array(blks_vlus) == ptn)]


rel_stuff = [list(g_usrs.vertex(g_usrs_vd[i]).out_edges()) for i in ptn_usrs]
rel_edges = list(itertools.chain.from_iterable(rel_stuff))

rel_edges2 = [(g_usrs_id[e.source()], g_usrs_id[e.target()], plcnt[e]) for e in rel_edges]

df_ptn = pd.DataFrame(rel_edges2, columns = ['usr', 'mbid', 'plcnt'])
df_ptn_grp = df_ptn[['mbid', 'plcnt']].groupby('mbid').sum()


# maybe vertex maps faster for subsetting? 
# also need to consider the weights

ptn_table = """
CREATE TEMPORARY TABLE ptn_table (
mbid String,
plcnt_ptn Int16
)
"""
client.execute('drop table ptn_table')
client.execute(ptn_table)

chnk_sz = 10000
chnk = []
c = 0
for i in df_ptn_grp.itertuples():

    chnk.append((i.Index, i.plcnt))
    c +=1
    if c == chnk_sz:
        client.execute('INSERT INTO ptn_table VALUES', chnk)
        chnk = []
        c = 0

client.execute('INSERT INTO ptn_table VALUES', chnk)


client.execute('select count(*) from ptn_table limit 1')
client.execute('select count(*) from fnl_qry limit 1')

ptn_dfc_qry = """SELECT * FROM fnl_qry JOIN ptn_table USING mbid"""

client.execute("SELECT * FROM fnl_qry limit 1")

client.execute("SELECT * FROM fnl_qry limit 1")


ptn_rows = client.execute(ptn_dfc_qry)

dont select cnt, just use ptn_clnt as cnt

dfc_ptn = pd.DataFrame(ptn_rows, columns = ['lfm_id','cnt', 'tag', 'weight', 'rel_weight',
                                               'artist', 'erl_rls', 'len_rls_lst'] + vrbls + ['ptn_plcnt'])

gnrs_ptn = list(np.unique(dfc_ptn['tag']))
# artsts = list(np.unique(dfc['artist']))
# trks = list(np.unique(dfc['lfm_id']))

acst_gnr_dict_ptn = dict_gnrgs(dfc_ptn, gnrs_ptn, pd)
for i in gnrs_ptn[0:30]:
    print(len(acst_gnr_dict_ptn[i]))

    # i have to rewrite so many functions that now rely on globals
    - gnr_sup_dicts
    # and what's it worth? 
    sz_dict, gnr_ind, waet_dict, vol_dict = gnrt_sup_dicts(acst_gnr_dict_ptn, gnrs_ptn)

    szs = [sz_dict[i] for i in gnrs_ptn]

    # should i drop genres under thresholds?
    # then it'd be easier to rewrite get_dfs take into only the links of each partition
    # am i doing circular reasoning? first have to create general dfc from which i than build specifics?

    # nope not really
    # users are just merged at the end, and don't have impact on dfc construction
    # guess dfc/fnl_qry_table could be seen as preparing the ground on which the different groups then battle it out
    # NOPE NOPE NOPE
    # if i subset early in dfc construction by only selecting the users in partition
    # i need two: because i still need to get the general dfc to get (the songs to get) the users in the first place

    # but that's no excuse to select from dfc for each partition if i care about gnr_criteria (size, concentration)
    # can gnr criteria even be the same with different partition sizes?
    # yeah think so, they're minimal requirements

    # but if i want to be sure they are adhered to i have to join people early
    # will probably also result in variation in gnrs?
    # could also be that sum of groups produces less than complete dfc if some borderline genres are spread -> not fulfilled in either
    # should i still keep them in? do they really exist? not really if i assume that classification systems live in the minds of the people in the partitions
    # then those split genres are like data artifacts i guess

    # does it even make sense then to use more users in total than i can put in a SBM?
    # would be sooo nice to have surfsara access but don't
    # but is question on its own
    # don't think so: dfc is just some agglomerate of different CS, has no substantive meaning
    

    el_ttl = gnrt_acst_el_mp(gnrs, 5)




# *** consistency/reliability check

csist_res = []

for i in range(6):

    t1 = time.time()
    # state = minimize_blockmodel_dl(g_usrs_1md, B_min = 4, B_max = 4)
    # state = minimize_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))


    state = minimize_blockmodel_dl(g_one_mode,
                                   # B_min = 3, B_max = 3,
                                   bisection_args = bisection_args,
                                   mcmc_args  = mcmc_args,
                                   mcmc_equilibrate_args = mcmc_equilibrate_args,
                                   shrink_args = shrink_args,
                                   deg_corr = True)
    
    t2 = time.time()
    print(t2-t1)

    blks = state.get_blocks()
    blks_vlus = [blks[i] for i in g_one_mode.vertices()]
    print(Counter(blks_vlus))
    
    unq_blks = np.unique(blks_vlus)
    run_res = []

    for k in unq_blks: run_res.append([])
    
    c = 0
    for x in blks_vlus:
        run_res[x].append(c)
        c+=1

    csist_res.append(run_res)

# for i in range(10):
#     csist_res.append(run_res)

all_grps_len = len(list(itertools.chain.from_iterable(csist_res)))

res_mat = np.zeros((all_grps_len,all_grps_len))


c1 = 0
for i in csist_res:
    for i2 in csist_res[csist_res.index(i)]:
        # print(len(i2))

        
        c2 = 0
        for k in csist_res:
            for k2 in csist_res[csist_res.index(k)]:
                ovlp = 2*len(set(i2) & set(k2))/(len(set(i2)) + len(set(k2)))
                # print(c1, c2, ovlp)
                res_mat[c1,c2] = ovlp
                c2+=1

        c1 +=1

# plt.matshow(res_mat)
# plt.show()
# how 

# from sklearn.cluster.bicluster import SpectralBiclustering

clust_mdl = SpectralBiclustering(n_clusters = 3)
clust1 = clust_mdl.fit(res_mat)

col_lbls = clust1.column_labels_
col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

row_lbls = clust1.row_labels_
row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
row_ord2 = list(itertools.chain.from_iterable(row_ord))

res_mat2 = res_mat[row_ord2,:][:,col_ord2]

# plt.matshow(res_mat)
plt.matshow(res_mat2)
plt.show()


# clearly 4 clusters, 3 or 5 produce nonsenical results
# i guess that counts as reliabilty?

# NOW NOT GOOD
# I think blocks are supposed to be of same size
# could also be that parameters in ptn_prep are wrong; still quite NOT GOOD when using default settings
# maybe directed links needed
# check 1k with varying cutoffs

# 1k, 0,04, 4 clusters: at least reliable
# but probably could cluster as well into 2: one well connected, one sparse connected
# yeah but it then muddles the distinctions to quite some degree
# 4 looks best tbh
# there is lack of overlap between two of non-sparse clusters

# might be that 0.04 produces more clear cut results than 0.02? 
# yup: 0.02 produces more internal high ones, but also more confusion, and large cluster only has 12 (not 15)

# i think speedy partitioning might be bad
# 0.02: one cluster very blurry
# 0.04: clusters not correctly extracted
# 0.04: slow

# WTH now (0.02, unmodified) no methods provides proper clustering?

# SIM graph provides very nice reliable clustering
# increasing speed reduces quality, still recognizable, but imo not worth the risk
# setting exact to True increases quality, but imo still worse than slow

# does number of edges matter?
# yup: 30k edges take about twice as long as 10k with almost same number of nodes
# minimizing number of edges would be nice
# high cutoffs
# just select N highest edges for each user
# maybe square? like if you have 100 edges  originally, you're allowed to keep 10, if 25, 5

# how to do it for each user but not keep it undirected?
# idk, i guess i'll just have to live with it? could adjust the strength of the sorting for each edge

# loop over edges and delete them with probability depending on source/target?
# keep or delete?
# don't like it it's so random

# where to sample
# - CH: nope doesn't work technically
# - in usr graph: technically works, but it feels a bit weird, also weird results
# - in two-mode graph: could sample till out_degree is under sqrt(outdegree)


el_smpl = []
for v in g_one_mode.vertices():
    v_nbrs = list(v.out_neighbors())

    max_nbrs = math.floor(np.sqrt(len(v_nbrs)) + 2)
    print(max_nbrs)
    

smpl_ep = g_one_mode.new_edge_property('bool')

for e in g_one_mode.edges():
    smpl_ep[e] = True

es = list(g_one_mode.edges())
random.shuffle(es)

for e in es:
    src = e.source()
    tgt = e.target()

    org_src_deg = src.out_degree()
    org_tgt_deg = tgt.out_degree()

    alctd_src_deg = math.floor(np.sqrt(org_src_deg)+2)
    alctd_tgt_deg = math.floor(np.sqrt(org_tgt_deg)+2)

    cur_src_deg = src.out_degree(smpl_ep)
    cur_tgt_deg = tgt.out_degree(smpl_ep)

    if cur_src_deg > alctd_src_deg  and cur_tgt_deg > alctd_tgt_deg:
        smpl_ep[e] = False
    
deg_org = []
deg_new = []

for v in g_one_mode.vertices():
    deg_org.append(v.out_degree())
    deg_new.append(v.out_degree(smpl_ep))


g_one_mode = Graph(GraphView(g_one_mode, efilt = smpl_ep), prune=True)


t1 = time.time()
statex = minimize_blockmodel_dl(g_one_mode2)
t2 = time.time()



# *** general clustering check

col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

row_lbls = clust1.row_labels_
row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
row_ord2 = list(itertools.chain.from_iterable(row_ord))



# *** reorder with SBM results

adj_mat = adjacency(GraphView(g_one_mode, directed=False)).toarray()

state = minimize_blockmodel_dl(g_one_mode,
                               B_min = 3, B_max = 3,
                               bisection_args = bisection_args,
                               mcmc_args  = mcmc_args,
                               mcmc_equilibrate_args = mcmc_equilibrate_args,
                               shrink_args = shrink_args,
                               deg_corr = True)

blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_one_mode.vertices()]
print(Counter(blks_vlus))
    
unq_blks = np.unique(blks_vlus)

col_lbls = row_lbls = blks_vlus

col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
row_ord2 = list(itertools.chain.from_iterable(row_ord))

res_mat2 = adj_mat[row_ord2,:][:,col_ord2]

# plt.matshow(adj_mat)
plt.matshow(res_mat2)
plt.show()



# ** SBM

# *** tut

import graph_tool as gt

g = gt.collection.data["football"]
state = minimize_blockmodel_dl(g)



# *** actual

# pointless, has to be put into 1 mode first, also allows more fine-tuning
# also weights work so not much information lost

# 

state2 = minimize_blockmodel_dl(g_usrs_flt)

state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight], rec_types=["discrete-binomial"]))

# *** hiearchical: expensive


tx1 = time.time()
state_hrc = minimize_nested_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))
tx2 = time.time()
# hiearchical takes so much longer
# 100: 21 sec
# 300, min common 20: 21
# don't want it anyways

# weights also expensive 
# tinkering with the features
state = minimize_blockmodel_dl(g_usrs_1md, B_min = 3, B_max = 6),
                               state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))


# ** AHC


dist_mat = -np.log(tri)
actual_max = np.max(dist_mat[np.where(dist_mat < math.inf)])
dist_mat[np.where(dist_mat > actual_max)] = actual_max + 2

nph(dist_mat)



from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters = 5, affinity='precomputed', linkage ='complete')
clstrs = cluster.fit_predict(dist_mat)
Counter(clstrs)

# use graph tool clustering? need to check how expensive
# for genre structure? think not: don't want latent communities, but hierachical order of existing nodes

https://graph-tool.skewed.de/static/doc/demos/inference/inference.html


# joining with fnl_qry basically halves the size
# memory consumption is not good at all

# ** some filtering
# kinda useless, rather need 1 mode transformation

sample_bin_vp = g_usrs.new_vertex_property('bool')

sample_nbrs = []
for i in sample_ids:
    sample_nbrs = sample_nbrs + [int(i) for i in g_usrs.vertex(i).out_neighbors()]

uniq_sample_nbrs = np.unique(sample_nbrs)

for i in sample_ids + list(uniq_sample_nbrs):
    sample_bin_vp[g_usrs.vertex(i)] = True


# g_usrs_flt = GraphView(g_usrs, vfilt = sample_bin_v)


# ** check the matrix thing for graphs generally with SpectralBiclustering

adj_mat = adjacency(GraphView(g_one_mode, directed=False)).toarray()
adj_clst = clust_mdl.fit(adj_mat)

col_lbls = adj_clst.column_labels_
col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

row_lbls = adj_clst.row_labels_
row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
row_ord2 = list(itertools.chain.from_iterable(row_ord))

plt.matshow(adj_mat)
plt.show()

# * scrap

# ** old dfc function

def get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
            min_unq_artss, tmax_propx1, max_propx2, d1, d2, 
            client, pd):
    # still has to be adopted to be able to accommodate time slices
    # wonder if the subsequent sorting can result in violations again?
    
    """retrieves acoustic data (vrbls) and tags of corresponding songs
        queries are written to only retrieve only complete matches (songs for which both musicological data and tags are available
    vrbls: variables for which to get musicological information
    min_cnt: minimal playcount for song
    min_weight: minimal absolute value for tagging to be included
    min_rel_weight: minimal relative value for tagging to be included
    min_tag_aprnc: minimal number of unique songs tag has to appear with
    min_unq_artsts: minimum number of unique artists for tag
    max_propx1: maximum percentage of songs in a genre by the largest artist
    max_propx2: maximum volume (rel_weight * cnt) in genre by largest artist
    """

    vrbl_strs  = ", ".join(vrbls)
    # TEST VALUES
    # min_weight = 10
    # min_rel_weight = 0.05
    # min_tag_aprnc = 5
    # min_cnt = 400

    # create merged df from beginning
    # try to split it the queries into strings
    # use temporary tables


    # gets the mbids that are can be used in terms of minimal playcount and acoustic data availability
    # basic = basis for further operations
    # probably should integrate temporal part here

    mbid_tbl_basic = """
    CREATE TEMPORARY TABLE mbids_basic
    (
    mbid_basic String,
    cnt Int16,
    artist String,
    erl_rls Int16,
    len_rls_lst Int8
    )
    """
    # d1 = '2011-10-01'
    # d2 = '2011-11-01'
    
    # filters by date
    date_str = """SELECT mbid, cnt FROM (
    SELECT song as abbrv, count(song) AS cnt FROM logs
        WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
        GROUP BY song
        HAVING cnt > """ + str(min_cnt) + """
    ) JOIN (
        SELECT * FROM song_info3) 
        USING abbrv"""


    mbid_basic_insert = """
    INSERT INTO mbids_basic 
    SELECT * FROM (
        SELECT lfm_id as mbid, cnt from acstb2
        JOIN
        ( """ + date_str + """ ) USING mbid
        ) JOIN ( SELECT lfm_id AS mbid, artist, erl_rls, len_rls_lst 
        FROM addgs ) USING mbid
    """

    # join with addgs
    
    
    # SELECT lfm_id AS mbid_basic, cnt, artist, erl_rls, len_rls_lst FROM addgs
    # JOIN mbids_basic USING mbid_basic

    # integrate lsnrs and plcnt from dones_tags?
    # does not vary over time...
    # is like looking in the future: has information from 2019 -> can't really use it to predict stuff in 2012
    



    # tags with basic requirement (in entire df)
    tag_tbl_basic = """
    CREATE TEMPORARY TABLE tags_basic
    ( tag_basic String)
    """

    tag_basic_insert = """
    INSERT INTO tags_basic
    SELECT tag
            FROM tag_sums 
            WHERE (weight > """ + str(min_weight) + """) 
            AND (rel_weight > """ + str(min_rel_weight) + """ )
            GROUP BY tag
            HAVING count(tag) > """ + str(min_tag_aprnc)


    # select the tags that correspond to the relevant songs, which is not useless
    basic_songs_tags_tbl = """
    CREATE TEMPORARY TABLE basic_songs_tags (
    mbid String,
    cnt Int16,
    tag String,
    weight Int8,
    rel_weight Float32, 
    artist String, 
    erl_rls Int16, 
    len_rls_lst Int8
    )
    """

    # select tags of songs that fulfil requirements generally (but maybe not in intersection)
    basic_songs_tags = """INSERT INTO basic_songs_tags
    SELECT mbid, cnt, tag, weight, rel_weight, artist, erl_rls, len_rls_lst
    FROM (
        SELECT mbid, tag, weight, rel_weight 
            FROM tag_sums

         JOIN (
            SELECT tag_basic as tag FROM tags_basic) 
        USING tag
        WHERE (weight > """ + str(min_weight) + """) 
        AND (rel_weight > """ + str(min_rel_weight) + """ ))

    JOIN (
        SELECT mbid_basic as mbid, cnt, artist, erl_rls, len_rls_lst from mbids_basic)
    USING mbid"""
    
    # get tags that are actually present enough in intsec
    # no real need for separate table for this, not that big an operation and only done once
    intsect_tags = """
    SELECT tag from (
        SELECT tag, cnt_tag, unq_artsts, max_cnt2, max_cnt2/cnt_tag as propx, max_sz2/szx as propx2 FROM (
            SELECT tag, count(tag) as cnt_tag, uniqExact(artist) as unq_artsts, sum(cnt*rel_weight) as szx
            FROM basic_songs_tags
            GROUP BY tag
            HAVING count(tag) > """ + str(min_tag_aprnc) + """ 
            AND uniqExact(artist) > """ + str(min_unq_artsts) + """
            ) 

        JOIN (
            SELECT tag, max(cnt2) as max_cnt2, max(sz2) as max_sz2 from (
                SELECT tag, artist, count(*) as cnt2, sum(cnt*rel_weight) as sz2
                FROM basic_songs_tags
                GROUP BY(tag, artist)
            )
        GROUP BY tag)
    USING tag
    )
    WHERE propx < """ + str(max_propx1) + """
    AND propx2 < """ + str(max_propx2)
    

    # boil down basic_songs_tags to intersection requirements
    int_sec_all = """
    SELECT * from basic_songs_tags
    JOIN ( """ + intsect_tags + """)
    USING tag"""
    
    # make merge table by getting stuff from acstb in
    # filtered on acstb before so should all be in there, and seems like it is

    fnl_qry_table = """
    CREATE TEMPORARY TABLE fnl_qry (
    mbid String,
    cnt Int16,
    tag String,
    weight Int8,
    rel_weight Float32, 
    artist String, 
    erl_rls Int16, 
    len_rls_lst Int8,
    """ + ",\n".join([i + ' Float32' for i in vrbls])+ ")"

    

    merge_qry = """
    INSERT INTO fnl_qry SELECT lfm_id as mbid, cnt, tag, weight, rel_weight, artist, erl_rls, len_rls_lst, """ + vrbl_strs + """ from acstb2
    JOIN (""" + int_sec_all + """) USING mbid"""
    
    drops = [
        'drop table mbids_basic',
        'drop table tags_basic',
        'drop table basic_songs_tags'
        'drop table fnl_qry']
    for d in drops:
        try:
            client.execute(d)
        except:
            pass
    
    client.execute(mbid_tbl_basic)
    client.execute(mbid_basic_insert)
    client.execute(tag_tbl_basic)
    client.execute(tag_basic_insert)
    client.execute(basic_songs_tags_tbl)
    client.execute(basic_songs_tags)
    client.execute(fnl_qry_table)
    client.execute(merge_qry)
    

    rows_merged = client.execute('SELECT * from fnl_qry')

    dfc2 = pd.DataFrame(rows_merged, columns = ['lfm_id','cnt', 'tag', 'weight', 'rel_weight',
                                               'artist', 'erl_rls', 'len_rls_lst'] + vrbls)
    # generate string for tag data

    
    return(dfc)


# ** clustering SBM  partition
# clustring without constraint and then clustering those 30+ with AHC results in largest position taking up 90+% of vertices
# *** cluster SBM partitions
# maybe group without limitation, and then group the clusters together in second turn?
# not really optimistic: shows the same pattern as the previous clustering attemps: gradient of well-connectedness
# that's kinda because of the usr-song distribution (exponential) which causes skewed usr degree distribution, with the voracious people being well connected

# ahhh fuckkkk
# possible explanations:
# - sample just fundamentally biased towards omnivores
# - too many items, weights: maybe just focus on top 1k songs, unweighted? 

usr_degs = [g_usrs.vertex(g_usrs_vd[i]).out_degree(plcnt) for i in unq_usrs]
# nph(usr_degs)

# degs = [i.out_degree() for i in GraphView(g_one_mode, directed=False).vertices()]
# nph(degs)


e = state.get_matrix().toarray()
plt.matshow(e)
plt.show()

# that's overlap mat
# sorenson dice similarity?

deg_vec = np.sum(e, axis=0)

deg_col_ar = np.array([deg_vec]*len(deg_vec))
deg_row_ar = deg_col_ar.T

deg_ar = deg_col_ar + deg_row_ar

sim_ar = 2*e/deg_ar

plt.matshow(dist_ar)
plt.show()

tual_min = np.min(sim_ar[np.where(sim_ar > 0)])

dist_ar = -np.log(sim_ar+tual_min/2)


cluster = AgglomerativeClustering(n_clusters = 3, affinity='precomputed', linkage ='average')
clstrs = cluster.fit_predict(dist_ar)
Counter(clstrs)

import scipy.cluster.hierarchy as sch

sch.dendrogram(sch.linkage(dist_ar, method='average'))
plt.show()


col_lbls = row_lbls = clstrs
unq_blks = list(np.unique(clstrs))

col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
row_ord2 = list(itertools.chain.from_iterable(row_ord))

res_mat2 = sim_ar[row_ord2,:][:,col_ord2]

plt.matshow(res_mat2)
plt.show()

# maybe just 3 partitions is best?
# *** get partition info back

clstrs_lrg = {}
for i in unq_blks:
    clstrs_lrg[i] = []

c = 0
for i in clstrs:
    clstrs_lrg[c] = i
    c +=1

blks_lrg = g_one_mode.new_vertex_property('int')
for v in g_one_mode.vertices():
    blks_lrg[v] = clstrs_lrg[blks[v]]

blks_lrg_vlus = [blks_lrg[blks[v]] for v in g_one_mode.vertices()]
# SO FUCKED RIGHT NOW

# ** old code, now optimized in other files

# get users
usr_string = """
SELECT usr, mbid, cnt FROM (
    SELECT * FROM (
        SELECT usr, song as abbrv, count(usr,song) as cnt FROM logs
            WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
            GROUP BY (usr,song)

        ) JOIN (
            SELECT mbid, abbrv FROM song_info3)
            USING abbrv
    ) JOIN (
        SELECT distinct(mbid) from fnl_qry
    ) USING mbid"""

usr_trk_lnks = client.execute(usr_string)

# don't like how long loading data into python takes...
# also the amount of memory jesus

unq_usrs = np.unique([i[0] for i in usr_trk_lnks])
unq_trks = np.unique([i[1] for i in usr_trk_lnks])


g_usrs = Graph()
plcnt = g_usrs.new_edge_property('int')

g_usrs_id = g_usrs.add_edge_list(usr_trk_lnks, hashed = True, string_vals = True, eprops = [plcnt])
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs_id)

# usr_trk_lnks = 0

N_SAMPLE = 4000

usrs_sample = sample(list(unq_usrs), N_SAMPLE)
sample_ids = [g_usrs_vd[i] for i in usrs_sample]

usr_cmps = list(itertools.combinations(sample_ids, 2))

tx1 = time.time()
smpl_sims = vertex_similarity(g_usrs, 'dice', vertex_pairs = usr_cmps, eweight = plcnt)
tx2 = time.time()
# 50k/sec
# 10k usrs would take 1k secs, seems sufficient? 
# where to split? if i restart CH server righ after getting the rows, i think i can put everything in one script

# how to put into np array

tri = np.zeros((N_SAMPLE, N_SAMPLE))
tri[np.triu_indices(N_SAMPLE, 1)] = smpl_sims

tri.T[np.triu_indices(N_SAMPLE, 1)] = smpl_sims

# get common matrix 
deg_vec = [g_usrs.vertex(i).out_degree(plcnt) for i in sample_ids]
deg_ar = np.array([deg_vec]*N_SAMPLE)
deg_ar2 = (deg_ar + np.array([deg_vec]*N_SAMPLE).transpose())/2

cmn_ar = deg_ar2*tri
ovlp_ar = cmn_ar/deg_ar

# one_mode1 = np.where(np.tril(cmn_ar) > 0)
one_mode_drct = np.where(ovlp_ar > 0.04)

elx = []

for i in zip(one_mode_drct[0], one_mode_drct[1]):
    nd1 = usrs_sample[i[0]]
    nd2 = usrs_sample[i[1]]
    vlu = ovlp_ar[i[0], i[1]]
    elx.append((nd1, nd2, vlu))
    
    # if vlu > 20:
    #     lnk = (usrs_sample[nd1], usrs_sample[nd2], vlu)
    #     elx.append(lnk)

g_usrs_1md = Graph()
g_usrs_1md_strng = g_usrs_1md.new_edge_property('double')

g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True, eprops = [g_usrs_1md_strng])
# g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True)


tx1 = time.time()
state = minimize_blockmodel_dl(g_usrs_1md, B_min = 4, B_max = 4)
# state = minimize_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))
tx2 = time.time()

blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_usrs_1md.vertices()]
print(Counter(blks_vlus))

# 321 sec for 1k
# 40 sec for 400
# 4 sec for 100
# 6 sec for 300 with min songs in common = 20
# 1k filtered (ovlp > 0.05), 20k edges (binary): 11 sec
# are weights expensive? 
# 1k filtered (ovlp > 0.05), weighted: 49-52
# seems that binary edges basically allow double number of users
# 1k users, 150k edges (ovlp > 0.01): 41 seconds
# 1k users, 150k edges (ovlp > 0.01), 3-6 blocks: 35 seconds
# 1k users, 150k edges (ovlp > 0.01), exactly 3 blocks: 35 seconds
# 1k users, 150k edges (ovlp > 0.01), exactly 4 blocks: 35 seconds: looks better in terms of block concentration:
# if cutoff is too high only the voracious ones will be connected, if it's too low it's too expensive
# 1k users, 76k edges ((ovlp > 0.02): 22 secs; blocks still somewhat equaly sized
# 1k users, 45k edges ((ovlp > 0.03): 16 secs; blocks still somewhat equaly sized
# 1k users, 28k edges ((ovlp > 0.04): 12 secs; blocks still somewhat equaly sized
# 1k users, 19k edges ((ovlp > 0.05): 9 secs; blocks still somewhat equaly sized
# like 0.04 better for some reason
# 4k users, 464k edges (ovlp > 0.04): 496 sec, probably because memory full
# nope not due to memory issue: closing everything else, still takes 480 secs
# memory consumption moderate over all, but seems to go up in the end

# annoy surfsara a bit more? now i could really use all that power
# JUST use 4/5k people? still bigger than most surveys tbh



# should outsource stuff to different scripts
# the gt operations are somewhat self-contained, so should be rather easy to outsource
# interface is main problem: how


e = state.get_matrix()
plt.matshow(e.todense())
plt.show()

blks = state.get_blocks()
blks_vlus = [blks[i] for i in g_usrs_1md.vertices()]
Counter(blks_vlus)

strngs = [g_usrs_1md_strng[e] for e in g_usrs_1md.edges()]
nph(strngs)

nph(np.log(strngs))


