
# what if i cluster first (the sample) without concern for acoustics
# then write the partition of each user to the usrs1k
# then call get_dfs with just those?

# means that the clusters will be based also on songs for which i don't have acoustic data
# seems not worse: if there's bias in the acoustic data, it will at least not be in cluster construction

# should be rewritten with either TEMPORARY TABLES or proper string composition 


sel_str ="""INSERT INTO el SELECT usr, song, count(usr,song) as cnt FROM (

        SELECT * FROM (
            SELECT usr, song, time_d FROM logs
            WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
        )

        JOIN (SELECT song, count(song) as song_cnt FROM logs
            WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
            GROUP BY song
            HAVING song_cnt > 10
            ) USING song

    ) JOIN (SELECT abbrv2 as usr from usrs4k)
            USING usr
            WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
            GROUP BY (usr,song)"""


client.execute('drop table el')
client.execute("CREATE TEMPORARY TABLE el (usr String, song String, cnt Int16)")
client.execute(sel_str)
client.execute("select count(*) from el")


usr_string = """
SELECT usr, song, cnt FROM (
    SELECT usr,song, cnt FROM el
    JOIN (SELECT song, count(song) as song_cnt2 FROM el
    GROUP BY song HAVING song_cnt2 > 5 )  USING song
) JOIN ( SELECT usr, count(usr) FROM (
    SELECT usr,song, cnt FROM el
        JOIN (SELECT song, count(song) as song_cnt2 FROM el
        GROUP BY song HAVING song_cnt2 > 5 )  USING song
)
GROUP BY usr HAVING count(usr) > 50) USING usr
"""

# * get users

usr_trk_lnks = client.execute(usr_string)

# don't like how long loading data into python takes...
# also the amount of memory Jesus
# focus only on most relevant songs reduces resources needed, also makes similarity calculations faster

unq_usrs = np.unique([i[0] for i in usr_trk_lnks])
unq_trks = np.unique([i[1] for i in usr_trk_lnks])

g_usrs = Graph()
g_usrs.edge_properties['plcnt'] = g_usrs.new_edge_property('int')

g_usrs.vp['id'] = g_usrs.add_edge_list(usr_trk_lnks, hashed = True, string_vals = True, eprops = [g_usrs.ep.plcnt])
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs.vp.id)
smpl_ep = g_usrs.new_edge_property('bool')

# x = [g_usrs.vertex(g_usrs_vd[v]).out_degree() for v in unq_usrs]

# song sampling
for u in unq_usrs:

    vu = g_usrs.vertex(g_usrs_vd[u])
    # u_dg_org = vu.out_degree(g_usrs.ep.plcnt)
    u_dg_org = vu.out_degree()
    # alctd_dg = u_dg_org**0.5
    alctd_dg = u_dg_org*0.25

    new_deg = 0
    
    sngs_el = list(vu.out_edges())
    random.shuffle(sngs_el)

    # filter CH query with user count
    for e in sngs_el:
        
        new_deg = new_deg + 1
        # g_usrs.ep.plcnt[e]
        smpl_ep[e] = True
        
        if new_deg > alctd_dg:
            break

g_usrs = Graph(GraphView(g_usrs, efilt = smpl_ep), prune=True)
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs.vp.id)

# N_SAMPLE = 4000
N_SAMPLE = len(unq_usrs)

usrs_sample = sample(list(unq_usrs), N_SAMPLE)
sample_ids = [g_usrs_vd[i] for i in usrs_sample]

usr_cmps = list(itertools.combinations(sample_ids, 2))

t1 = time.time()
smpl_sims = vertex_similarity(g_usrs, 'dice', vertex_pairs = usr_cmps, eweight = g_usrs.ep.plcnt)
t2 = time.time()
# 50k/sec
# weird, with US 1k sample it's down to 25k/sec
# 10k usrs would take 1k secs, seems sufficient? 
# where to split? if i restart CH server righ after getting the rows, i think i can put everything in one script



# can also just use cutoff for smpl_sims? 
# usr_lnks_sim = np.where(smpl_sims > 0.0253)
# x = set(usr_lnks_sim[0]) - set(usr_lnks[0]) # not exactly the same edges, 1500/10k not there
usr_lnks = np.where(smpl_sims > 0.02)

elx = []
for i in usr_lnks[0]:
    rel_e =usr_cmps[i]
    elx.append((g_usrs_vd_rv[rel_e[0]], g_usrs_vd_rv[rel_e[1]]))
    

g_usrs_1md = Graph(directed=False)

g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True)
# g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True)

g_usrs_1md.vertex_properties['id'] = g_usrs_1md_id

# g_usrs_1md.save('one_mode1k_dice_005.gt')
# g_usrs_1md.save('one_mode1k_dice_0025_smpl.gt')
g_usrs_1md.save('one_mode4k_dice_002_smpl_l025.gt')


# NEED WAY TO PASS VARIABLES
# maybe something like a config_dict

# ptn prep and ptning has to run before get_dfs in acst_hier
# hm not clear how memory is looking after running feature extraction





# * scrap
# different get_dfc function that only differs in using additional temporary table that stores the results of the final query
# idea as to select different sections
# but if i take multiple CS seriously i can't use aggregatve and filter that, but have to built separate aggregates for each CS


# def get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
#             min_unq_artsts, max_propx1, max_propx2, d1, d2, 
#             client, pd):
#     # still has to be adopted to be able to accommodate time slices
#     # wonder if the subsequent sorting can result in violations again?
    
#     """retrieves acoustic data (vrbls) and tags of corresponding songs
#         queries are written to only retrieve only complete matches (songs for which both musicological data and tags are available
#     vrbls: variables for which to get musicological information
#     min_cnt: minimal playcount for song
#     min_weight: minimal absolute value for tagging to be included
#     min_rel_weight: minimal relative value for tagging to be included
#     min_tag_aprnc: minimal number of unique songs tag has to appear with
#     min_unq_artsts: minimum number of unique artists for tag
#     max_propx1: maximum percentage of songs in a genre by the largest artist
#     max_propx2: maximum volume (rel_weight * cnt) in genre by largest artist
#     """

#     vrbl_strs  = ", ".join(vrbls)
#     # TEST VALUES
#     # min_weight = 10
#     # min_rel_weight = 0.05
#     # min_tag_aprnc = 5
#     # min_cnt = 400

#     # create merged df from beginning
#     # try to split it the queries into strings
#     # use temporary tables


#     # gets the mbids that are can be used in terms of minimal playcount and acoustic data availability
#     # basic = basis for further operations
#     # probably should integrate temporal part here

#     mbid_tbl_basic = """
#     CREATE TEMPORARY TABLE mbids_basic
#     (
#     mbid_basic String,
#     cnt Int16,
#     artist String,
#     erl_rls Int16,
#     len_rls_lst Int8
#     )
#     """
#     # d1 = '2011-10-01'
#     # d2 = '2011-11-01'
    
#     # filters by date
#     date_str = """SELECT mbid, cnt FROM (
#     SELECT song as abbrv, count(song) AS cnt FROM logs
#         WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
#         GROUP BY song
#         HAVING cnt > """ + str(min_cnt) + """
#     ) JOIN (
#         SELECT * FROM song_info3) 
#         USING abbrv"""


#     mbid_basic_insert = """
#     INSERT INTO mbids_basic 
#     SELECT * FROM (
#         SELECT lfm_id as mbid, cnt from acstb2
#         JOIN
#         ( """ + date_str + """ ) USING mbid
#         ) JOIN ( SELECT lfm_id AS mbid, artist, erl_rls, len_rls_lst 
#         FROM addgs ) USING mbid
#     """

#     # join with addgs
    
    
#     # SELECT lfm_id AS mbid_basic, cnt, artist, erl_rls, len_rls_lst FROM addgs
#     # JOIN mbids_basic USING mbid_basic

#     # integrate lsnrs and plcnt from dones_tags?
#     # does not vary over time...
#     # is like looking in the future: has information from 2019 -> can't really use it to predict stuff in 2012
    



#     # tags with basic requirement (in entire df)
#     tag_tbl_basic = """
#     CREATE TEMPORARY TABLE tags_basic
#     ( tag_basic String)
#     """

#     tag_basic_insert = """
#     INSERT INTO tags_basic
#     SELECT tag
#             FROM tag_sums 
#             WHERE (weight > """ + str(min_weight) + """) 
#             AND (rel_weight > """ + str(min_rel_weight) + """ )
#             GROUP BY tag
#             HAVING count(tag) > """ + str(min_tag_aprnc)


#     # select the tags that correspond to the relevant songs, which is not useless
#     basic_songs_tags_tbl = """
#     CREATE TEMPORARY TABLE basic_songs_tags (
#     mbid String,
#     cnt Int16,
#     tag String,
#     weight Int8,
#     rel_weight Float32, 
#     artist String, 
#     erl_rls Int16, 
#     len_rls_lst Int8
#     )
#     """

#     # select tags of songs that fulfil requirements generally (but maybe not in intersection)
#     basic_songs_tags = """INSERT INTO basic_songs_tags
#     SELECT mbid, cnt, tag, weight, rel_weight, artist, erl_rls, len_rls_lst
#     FROM (
#         SELECT mbid, tag, weight, rel_weight 
#             FROM tag_sums

#          JOIN (
#             SELECT tag_basic as tag FROM tags_basic) 
#         USING tag
#         WHERE (weight > """ + str(min_weight) + """) 
#         AND (rel_weight > """ + str(min_rel_weight) + """ ))

#     JOIN (
#         SELECT mbid_basic as mbid, cnt, artist, erl_rls, len_rls_lst from mbids_basic)
#     USING mbid"""
    
#     # get tags that are actually present enough in intsec
#     # no real need for separate table for this, not that big an operation and only done once
#     intsect_tags = """
#     SELECT tag from (
#         SELECT tag, cnt_tag, unq_artsts, max_cnt2, max_cnt2/cnt_tag as propx, max_sz2/szx as propx2 FROM (
#             SELECT tag, count(tag) as cnt_tag, uniqExact(artist) as unq_artsts, sum(cnt*rel_weight) as szx
#             FROM basic_songs_tags
#             GROUP BY tag
#             HAVING count(tag) > """ + str(min_tag_aprnc) + """ 
#             AND uniqExact(artist) > """ + str(min_unq_artsts) + """
#             ) 

#         JOIN (
#             SELECT tag, max(cnt2) as max_cnt2, max(sz2) as max_sz2 from (
#                 SELECT tag, artist, count(*) as cnt2, sum(cnt*rel_weight) as sz2
#                 FROM basic_songs_tags
#                 GROUP BY(tag, artist)
#             )
#         GROUP BY tag)
#     USING tag
#     )
#     WHERE propx < """ + str(max_propx1) + """
#     AND propx2 < """ + str(max_propx2)
    

#     # boil down basic_songs_tags to intersection requirements
#     int_sec_all = """
#     SELECT * from basic_songs_tags
#     JOIN ( """ + intsect_tags + """)
#     USING tag"""
    
#     # make merge table by getting stuff from acstb in
#     # filtered on acstb before so should all be in there, and seems like it is

#     fnl_qry_table = """
#     CREATE TEMPORARY TABLE fnl_qry (
#     mbid String,
#     cnt Int16,
#     tag String,
#     weight Int8,
#     rel_weight Float32, 
#     artist String, 
#     erl_rls Int16, 
#     len_rls_lst Int8,
#     """ + ",\n".join([i + ' Float32' for i in vrbls])+ ")"


#     merge_qry = """
#     INSERT INTO fnl_qry SELECT lfm_id as mbid, cnt, tag, weight, rel_weight, artist, erl_rls, len_rls_lst, """ + vrbl_strs + """ from acstb2
#     JOIN (""" + int_sec_all + """) USING mbid"""
    
#     drops = [
#         'drop table mbids_basic',
#         'drop table tags_basic',
#         'drop table basic_songs_tags'
#         'drop table fnl_qry']
#     for d in drops:
#         try:
#             client.execute(d)
#         except:
#             pass
    
#     client.execute(mbid_tbl_basic)
#     client.execute(mbid_basic_insert)
#     client.execute(tag_tbl_basic)
#     client.execute(tag_basic_insert)
#     client.execute(basic_songs_tags_tbl)
#     client.execute(basic_songs_tags)
#     client.execute(fnl_qry_table)
#     client.execute(merge_qry)
#     rows_merged = client.execute('SELECT * from fnl_qry')

#     dfc2 = pd.DataFrame(rows_merged, columns = ['lfm_id','cnt', 'tag', 'weight', 'rel_weight',
#                                                'artist', 'erl_rls', 'len_rls_lst'] + vrbls)
#     # generate string for tag data

    
#     return(dfc)

# dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
#               min_unq_artsts, max_propx1, max_propx2, d1, d2,
#               client, pd)

## ** debug how different thresholds effect edge and v count

v_cnts = []
e_cnts = []
for thd in np.arange(0, 0.05, 0.0005):
    print(thd)

    # use amount of links 
    usr_lnks = np.where(l_mins > thd)

    # use dice similarity
    # usr_lnks = np.where(smpl_sims > thd)

    elx = []
    for i in usr_lnks[0]:
        rel_e =usr_cmps[i]
        elx.append((g_usrs_vd_rv[rel_e[0]], g_usrs_vd_rv[rel_e[1]]))

    g_usrs_1md = Graph()

    g_usrs_1md_id = g_usrs_1md.add_edge_list(elx, hashed=True, string_vals=True)
    v_cnt = len(list(g_usrs_1md.vertices()))
    e_cnt = len(list(g_usrs_1md.edges()))
    v_cnts.append(v_cnt)
    e_cnts.append(e_cnt)

npl(v_cnts)
npl(e_cnts)






# ** directed links
# one_mode_drct = np.where(ovlp_ar > 0.04) 

# elx = []

# for i in zip(one_mode_drct[0], one_mode_drct[1]):
#     nd1 = usrs_sample[i[0]]
#     nd2 = usrs_sample[i[1]]
#     vlu = ovlp_ar[i[0], i[1]]
#     elx.append((nd1, nd2, vlu))
    
#     # if vlu > 20:
#     #     lnk = (usrs_sample[nd1], usrs_sample[nd2], vlu)
#     #     elx.append(lnk)

# print(len(elx)/N_SAMPLE**2)


# ** old usr similarity calculation that uses ovlp, is more parsimonious to just use DICE sim
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

# would be best to 'melt' the ovlp array into long lists
# maybe add small stuff to be able to select

mod_ovlp_ar = ovlp_ar+0.0123

l1 = mod_ovlp_ar[np.where(np.triu(mod_ovlp_ar, k=1) > 0)] - 0.0123
l2 = mod_ovlp_ar.T[np.where(np.triu(mod_ovlp_ar.T, k=1) > 0)] - 0.0123

l_ar = np.concatenate([l1[:,None],l2[:,None]], axis=1)

pos_mins = np.argmin(l_ar, axis=1)

l_mins = l_ar[np.arange(l_ar.shape[0]),pos_mins]

usr_lnks = np.where(l_mins > 0.02)

# ** old usr string
# usr_string = """
# SELECT usr, mbid, cnt FROM (
#     SELECT * FROM (
#         SELECT usr, song as abbrv, count(usr,song) as cnt FROM logs
#             WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
#             GROUP BY (usr,song)

#         ) JOIN (
#             SELECT mbid, abbrv FROM song_info3)
#             USING abbrv
#     ) JOIN (
#         SELECT distinct(mbid) from fnl_qry
#     ) USING mbid"""
