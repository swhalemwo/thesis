

def get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
            min_unq_artsts, max_propx1, max_propx2, d1, d2, 
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

usrs = [i[0] for i in usr_trk_lnks]
unq_usrs = np.unique(usrs)

g_usrs = Graph()
plcnt = g_usrs.new_edge_property('int')

g_usrs_id = g_usrs.add_edge_list(usr_trk_lnks, hashed = True, string_vals = True, eprops = [plcnt])
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs_id)

N_SAMPLE = 100

usrs_sample = sample(list(unq_usrs), N_SAMPLE)
sample_ids = [g_usrs_vd[i] for i in usrs_sample]


usr_cmps = list(itertools.combinations(sample_ids, 2))

smpl_sims = vertex_similarity(g_usrs, 'dice', vertex_pairs = usr_cmps, eweight = plcnt)


# how to put into np array

tri = np.zeros((N_SAMPLE, N_SAMPLE))
tri[np.triu_indices(N_SAMPLE, 1)] = smpl_sims

tri.T[np.triu_indices(N_SAMPLE, 1)] = smpl_sims

# get common matrix 
deg_vec = [g_usrs.vertex(i).out_degree(plcnt) for i in sample_ids]
deg_ar = np.array([deg_vec]*N_SAMPLE)
deg_ar2 = (deg_ar + np.array([deg_vec]*N_SAMPLE).transpose())/2


cmn_ar = deg_ar2*tri
one_mode1 = np.where(np.tril(cmn_ar) > 0)

elx = []

for i in zip(one_mode1[0], one_mode1[1]):
    nd1 = i[0]
    nd2 = i[1]
    
    lnk = (usrs_sample[nd1], usrs_sample[nd2], cmn_ar[nd1, nd2])
    elx.append(lnk)

g_usrs_1md = Graph(directed=False)
g_usrs_1md_strng = g_usrs_1md.new_edge_property('int')

g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True, eprops = [g_usrs_1md_strng])

tx1 = time.time()
state = minimize_blockmodel_dl(g_usrs_1md, state_args=dict(recs=[g_usrs_1md_strng], rec_types=["real-exponential"]))
tx2 = time.time()


# 40 sec for 400
# 4 sec for 100

# look at settings

e = state.get_matrix()
plt.matshow(e.todense())
plt.show()




strngs = [g_usrs_1md_strng[e] for e in g_usrs_1md.edges()]
nph(strngs)


# could translate into 1 mode graph by getting number of common vertices
# would have to binarize edges tho
# probably with rather high cutoff to not get that many edges -> rather clear clusters


## ** SBM

## *** tut

import graph_tool as gt

g = gt.collection.data["football"]
state = minimize_blockmodel_dl(g)



## *** real

# pointless, has to be put into 1 mode first, also allows more fine-tuning
# also weights work so not much information lost

state2 = minimize_blockmodel_dl(g_usrs_flt)


state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight], rec_types=["discrete-binomial"]))


## ** AHC

dist_mat = -np.log(tri)
actual_max = np.max(dist_mat[np.where(dist_mat < math.inf)])
dist_mat[np.where(dist_mat > actual_max)] = actual_max + 2

nph(dist_mat)



from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(distance_threshold = 5, n_clusters = None, affinity='precomputed', linkage ='complete')
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


g_usrs_flt = GraphView(g_usrs, vfilt = sample_bin_vp)
