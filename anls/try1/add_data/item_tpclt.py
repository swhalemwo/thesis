# * libs

import csv
import json
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# * funcs

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


client = Client(host='localhost', password='anudora', database='frrl')

vrbls = ['dncblt','gender','timb_brt','tonal','voice']

def get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc):
    # still has to be adopted to be able to accommodate time slices
    # wonder if the subsequent sorting can result in violations again?
    
    """retrieves acoustic data (vrbls) and tags of corresponding songs
        queries are written to only retrieve only complete matches (songs for which both musicological data and tags are available
    vrbls: variables for which to get musicological information
    min_cnt: minimal playcount for song
    min_weight: minimal absolute value for tagging to be included
    min_rel_weight: minimal relative value for tagging to be included
    min_tag_aprnc: minimal number of times tag has to appear
    """

    vrbl_strs  = ", ".join(vrbls)
    # TEST VALUES
    # min_weight = 10
    # min_rel_weight = 0.05
    # min_tag_aprnc = 5
    # min_cnt = 400

    # another try: try to create merged df from beginning
    # try to split it the queries into strings
    # use temporary tables

    # gets the mbids that are can be used in terms of minimal playcount and acoustic data availability
    # basic = basis for further operations
    mbid_tbl_basic = """
    CREATE TEMPORARY TABLE mbids_basic
    (
    mbid_basic String
    )
    """

    mbid_basic_insert = """
    INSERT INTO mbids_basic 
    SELECT lfm_id as mbid from acstb
    JOIN
    (
        SELECT mbid from song_info3
        WHERE cnt > """ + str(min_cnt) + """
    ) USING mbid
    """
    print(mbid_basic_insert)
    
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

    print(tag_basic_insert)
    client.execute('drop table tags_basic')
    client.execute(tag_tbl_basic)    
    client.execute(tag_basic_insert)

    # select the tags that correspond to the relevant songs
    # actually might be that the select of basic tags is quite useless and it's faster to just go over the relevant songs?
    # but still might need it for general tag filtering, and is not expensive
    
    basic_songs_tags_tbl = """
    CREATE TEMPORARY TABLE basic_songs_tags (
    mbid String,
    tag String,
    weight Int8,
    rel_weight Float32)
    """
    client.execute('drop table basic_songs_tags')
    client.execute(basic_songs_tags_tbl)

    # select tags of songs that fulfil requirements generally (but maybe not in intersection)
    basic_songs_tags = """INSERT INTO basic_songs_tags
    SELECT mbid, tag, weight, rel_weight 
    FROM (
        SELECT mbid, tag, weight, rel_weight 
            FROM tag_sums

         JOIN (
            SELECT tag_basic as tag FROM tags_basic) 
        USING tag
        WHERE (weight > """ + str(min_weight) + """) 
        AND (rel_weight > """ + str(min_rel_weight) + """ ))

    JOIN (
        SELECT mbid_basic as mbid from mbids_basic)
    USING mbid"""
    # print(basic_songs_tags)

    
    # get tags that are actually present enough in intsec
    # no real need for separate table for this, not that big an operation and only done once
    intsect_tags = """SELECT tag FROM basic_songs_tags
    GROUP BY tag
    HAVING count(tag) > """ + str(min_tag_aprnc)

    # boil down basic_songs_tags to intersection requirements

    int_sec_all = """
    SELECT * from basic_songs_tags
    JOIN ( """ + intsect_tags + """)
    USING tag"""
    print(int_sec_all)
    

    # make merge table by getting stuff from acstb in
    # filtered on acstb before so should all be in there, and seems like it is

    merge_qry = """
    SELECT lfm_id as mbid, """ + vrbl_strs + """ from acstb
    JOIN (""" + int_sec_all + """) USING mbid"""
    
    
    rows_tags = client.execute(client_str2)
    rows_tags = client.execute(con_sel)
    df_tags = pd.DataFrame(rows_tags, columns=['lfm_id', 'tag', 'rel_weight', 'cnt'])
    df_tags = pd.DataFrame(rows_tags, columns=['lfm_id', 'tag', 'cnt', 'rel_weight'])
    songs_tags = df_tags['lfm_id']
    len(np.unique(songs_tags))

    
    # generate string for tag data
    client_str2 ="""
    SELECT mbid, tag, rel_weight, cnt
    FROM (
        SELECT * FROM (
            SELECT * FROM tag_sums
            JOIN
            (
                SELECT tag, count(*) AS tag_ap FROM tag_sums
                WHERE weight > """ + str(min_weight) + """
                AND rel_weight > """ + str(min_rel_weight) + """
                GROUP BY tag
                HAVING tag_ap > """ + str(min_tag_aprnc) + """
            ) USING (tag)
        )
        JOIN
        (
            SELECT lfm_id as mbid
            FROM acstb
        ) USING (mbid)
    )
    JOIN
    (
        SELECT mbid, cnt
        FROM song_info3
        WHERE cnt > """ + str(min_cnt) + """
    ) USING mbid
    WHERE weight > """ + str(min_weight) + """ 
    AND rel_weight > """ + str(min_rel_weight)

    rows_tags = client.execute(client_str2)
    df_tags = pd.DataFrame(rows_tags, columns=['lfm_id', 'tag', 'rel_weight', 'cnt'])
    songs_tags = df_tags['lfm_id']
    len(np.unique(songs_tags))

    return(df_acst, df_tags)

min_cnt = 200
min_weight = 10
min_rel_weight = 0.1
min_tag_aprnc = 50

df_acst, df_tags = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc)
gnrs = list(np.unique(df_tags['tag']))

# seems to be working
# songs_acst = df_ac['lfm_id']
# songs_tags = df_tg['lfm_id']
# songs_tbp = list(set(songs_acst) - set(songs_tags))

def dict_gnrgs(df_acst, df_tags):
    """generates genre song dict,  acoustic position dict, tag row dict"""
    # make position dicts because dicts good
    # or rather, they would be if pandas subsetting wouldnt be slow AF
    # solution: make genre dfs only once in initialization

    gnr_song_dict = {} # lfm_ids of songs in genre
    tag_row_dict = {} # indices of tags in tags_df for each genre
    # gnr_tags_dict = {} # 
    
    for r in df_tags.itertuples():
        gnr = r.tag
        
        if gnr in gnr_song_dict.keys():
            gnr_song_dict[gnr].append(r.lfm_id)
            tag_row_dict[gnr].append(r.Index)
            # gnr_tags_dict[gnr].append(list(r))
            
        else:
            gnr_song_dict[gnr] = [r.lfm_id]
            tag_row_dict[gnr] = [r.Index]
            # gnr_tags_dict[gnr] = [list(r)]

    acst_pos_dict = {}

    for r in df_acst.itertuples():
        acst_pos_dict[r.lfm_id] = r.Index


    # merge tag and acst df, distribute those acst into dict with their respective genres
    df_mrgd = pd.merge(df_tags, df_acst, on='lfm_id')

    acst_gnr_dict = {}
    for r in df_mrgd.itertuples():
        gnr = r.tag

        if gnr in acst_gnr_dict.keys():
            acst_gnr_dict[gnr].append(r)
        else:
            acst_gnr_dict[gnr] = [r]

    for gnr in acst_gnr_dict.keys():
        acst_gnr_dict[gnr] = pd.DataFrame(acst_gnr_dict[gnr])

    # might be that previous dicts have become useless now, gotta clean up
    # get_df_cbmd is basically useless now
    return(gnr_song_dict, acst_pos_dict, tag_row_dict, acst_gnr_dict)


gnr_song_dict, acst_pos_dict, tag_row_dict, acst_gnr_dict = dict_gnrgs(df_acst, df_tags)


def get_inv_mat(df_gnr_cmbd, vrbls):
    """calculates the inverse covariance matrix used for mahalanobis distance, weighted by counts"""
    mat = df_gnr_cbmd[vrbls]

    aweits = df_gnr_cbmd['rel_weight']

    cov_mat = np.cov(mat, rowvar=0, aweights=aweits)
    cov_mat_inv = np.linalg.inv(cov_mat)
    return(cov_mat_inv)

def get_df_cbmd(gnr):
    """constructs combined df out of acoustic and tag df"""
    # gnr = 'rap'


    gnr_acst_ids = [acst_pos_dict[i] for i in gnr_song_dict[gnr]]

    # df_gnr_tags = df_tags[df_tags['tag']==gnr]
    # df_gnr_tags = df_tags.loc[tag_row_dict[gnr]]
    timeit(stmt='df_gnr_tags = df_tags.iloc[tag_row_dict[gnr]]')
    timeit(stmt='df_tags.iloc[tag_row_dict[gnr]]', globals=globals(), number=10)

        
    df_gnr_acst = df_acst.iloc[gnr_acst_ids]

    df[df.index.isin([1,3])]

    ar_tags = np.array(df_tags)
    timeit('ar_gnr_tags = ar_tags[tag_row_dict[gnr]]', globals=globals(), number=10)

    import pyximport
    import importlib

    importlib.reload(artest)
    artest.ar_test()
    


    t1 = time.time()
    df_gnr_cbmd = pd.merge(df_gnr_tags, df_gnr_acst, on='lfm_id')
    t2 = time.time()        

    return(df_gnr_cbmd)

# * playground

dfc = get_df_cbmd('rap')

tx = '047a94b1-c42e-4867-8aa5-36d8c1f57d00'
tx_loc = list(dfc['lfm_id']).index(trackx)

vrbls = ['dncblt','gender', 'timb_brt','tonal', 'voice']

## ** mahalanobis dist

# still not clear about how to weigh, whether to use weight or count arrrrr
# hmm original idea was to use cnts and use batches based on weights

slx, ovrl = [], []

for i in vrbls:
    print(i)
    slx.append(float(dfc[dfc['lfm_id'] == trackx][i]))
    ovrl.append(float(weighted_avg_and_std(dfc[i], dfc['cnt'])[0]))

cov_mat_inv = get_inv_mat(dfc, vrbls)

mahalanobis(slx, ovrl, cov_mat_inv)

dfc['mah_dist']=0.0

unq_trks = list(dfc['lfm_id'])

c = 0
for tx in unq_trks:
    tx_vls = dfc[dfc['lfm_id'] == tx][vrbls]
    mah_dist = mahalanobis(tx_vls, ovrl, cov_mat_inv)

    dfc.at[c, 'mah_dist'] = mah_dist
    c+=1

m_dists = dfc['mah_dist']
plt.hist(m_dists, bins='auto')
plt.show()

# ** why is this so high? need to test with normally distributed data
# could be due to binominal distribution..
# but still, that really no one is close is weird


rnd1 = np.random.normal(loc=3, scale=0.5, size = 1000)
rnd2 = np.random.normal(loc=2, scale=2, size = 1000)
rnd3 = np.random.normal(loc=4, scale=1, size = 1000)

rnd_ar = np.array([rnd1, rnd2, rnd3]).transpose()

cov_mat2 =np.cov(rnd_ar, rowvar=0)
cm_inv2 = np.linalg.inv(cov_mat2)

df_rnd = pd.DataFrame(rnd_ar, columns = ['r1', 'r2', 'r3'])

means = [np.mean(i) for i in [rnd1, rnd2, rnd3]]

mah_dists = []
for i in range(1000):
    distx = mahalanobis(rnd_ar[i,], means, cm_inv2)
    mah_dists.append(distx)
# hm still not many at 0, but much closer (min 0.1 vs ~0.8)


# ** data exploration
for v in vrbls:
    plt.subplot(5, 1, vrbls.index(v)+1)
    plt.hist(df_acst[v], bins='auto')

plt.show()
# fuck, basically bimodals
# not really sure if it makes sense to calculate mean of those


vrbls2 = ["mood_acoustic","mood_aggressive","mood_electronic","mood_happy","mood_party","mood_relaxed","mood_sad"]

# *** biased cov mat estimation
# see if i can make columns/observations less important 


rnd1 = np.random.normal(loc=0, scale=0.5, size = 1000)
rnd2 = np.random.normal(loc=0, scale=2, size = 1000)
rnd3 = np.random.normal(loc=0, scale=1, size = 1000)

rnd_ar = np.array([rnd1, rnd2, rnd3]).transpose()

row_weits = choices([i for i in range(10)], k=1000)

cov_mat_bs = np.cov(rnd_ar, rowvar = 0)
cov_mat_bs2 = np.cov(rnd_ar, rowvar = 0, aweights = row_weits)

W_rows = [[1,0.5,0.1]]*1000
W = np.array(W_rows)
Q = rnd_ar

QW = Q*W
C = QW.T.dot(QW)/W.T.dot(W)


Q[:,0].dot(Q[:,1])

# need zero mean? np my dude
# still not good: if i weigh entire column, it gets equalized because weights are in both parts of the equation

# just multiply column (by degree of normality violation) to change its SD? 
rnd_ar2 = rnd_ar*W

C3 = np.cov(rnd_ar2, rowvar=0)
# there's certainly a difference now LEL
# not so sure if i really want that tho
# but i might
# basically shrinks the variation in the dimensions i don't like

# *** measure extent of bimodality
def get_bmdl_dist(sd):
    v1_1 = np.random.normal(loc=1, scale = sd, size = 10000)
    v1_11 = [i for i in v1_1 if i < 1 and i > 0]
    v1_2 = np.random.normal(loc=0, scale = sd, size = 10000)
    v1_21 = [i for i in v1_2 if i > 0 and i < 1]

    v1 = random.sample(v1_11, 500) + random.sample(v1_21, 500)
    return(v1)

v1 = get_bmdl_dist(0.01)

plt.hist(v1, bins=30)
plt.show()

from scipy.stats import kurtosis

kurtosis(get_bmdl_dist(0.01))
kurtosis(get_bmdl_dist(1))
kurtosis(get_bmdl_dist(2))
kurtosis(np.random.normal(loc=1, scale=1, size=10000))

# seems reasonably to weight by kurtosis as measure of bimodality
# \cite{Darlington_1970_kurtosis}

import diptest


x2 = [float(i) for i in x]
x3 = np.array(x2)
diptest.dip(np.array(get_bmdl_dist(1)))
diptest.dip(np.array(get_bmdl_dist(0.1)))
diptest.dip(np.array(get_bmdl_dist(0.01)))

diptest.dip(np.random.normal(loc=1, scale=1, size=10000))

# i hope the dip statistic is not some kind of pvalue, but not clera





# idk if that's producing what i want

A = np.einsum('ki,kj->ij', Q*W, Q*W)
B = np.einsum('ki,kj->ij', W, W)
C = A/B


C2 = np.array([[ 1.  ,  0.1 ,  0.2 ], # set this beforehand, to test whether 
           [ 0.1 ,  0.5 ,  0.15], # we get the correct result
           [ 0.2 ,  0.15,  0.75]])

Q = np.array([[-0.6084634 ,  0.16656143, -1.04490324],
           [-1.51164337, -0.96403094, -2.37051952],
           [-0.32781346, -0.19616374, -1.32591578],
           [-0.88371729,  0.20877833, -0.52074272],
           [-0.67987913, -0.84458226,  0.02897935],
           [-2.01924756, -0.51877396, -0.68483981],
           [ 1.64600477,  0.67620595,  1.24559591],
           [ 0.82554885,  0.14884613, -0.15211434],
           [-0.88119527,  0.11663335, -0.31522598],
           [-0.14830668,  1.26906561, -0.49686309]])

W = np.array([[ 1.01133857,  0.91962164,  1.01897898],
           [ 1.09467975,  0.91191381,  0.90150961],
           [ 0.96334661,  1.00759046,  1.01638749],
           [ 1.04827001,  0.95861001,  1.01248969],
           [ 0.91572506,  1.09388218,  1.03616461],
           [ 0.9418178 ,  1.07210878,  0.90431879],
           [ 1.0093642 ,  1.00408472,  1.07570172],
           [ 0.92203074,  1.00022631,  1.09705542],
           [ 0.99775598,  0.01000000,  0.94996408],
           [ 1.02996389,  1.01224303,  1.00331465]])


# ** test with entire data as basis for cov mat

mat = df_acst[vrbls]

cov_mat = np.cov(mat, rowvar=0)
cov_mat_inv = np.linalg.inv(cov_mat)


c = 0
for tx in unq_trks:
    tx_vls = df_gnr_cbmd[df_gnr_cbmd['lfm_id'] == tx][vrbls]
    mah_dist = mahalanobis(tx_vls, ovrl, cov_mat_inv)

    df_gnr_cbmd.at[c, 'mah_dist'] = mah_dist
    c+=1

t1 = time.time()
for i in range(100000):
    mah_dist = mahalanobis(tx_vls, ovrl, cov_mat_inv)
t2 = time.time()    

# need to get playcount into weights 
# i'm not sure if using the entire data set is the right thing to do:
# importance of dimensions can differ between genres
# i think covariance has to be estimated for each genre
# if there's no difference it doesn't harm splitting anyways

# ** genre variation in mean/sds

# not actually sure what i need merge for? 
# dfc2 = pd.merge(df_tags[['lfm_id', 'cnt']], df_acst , on ='lfm_id')
# dfc3 = dfc2.drop_duplicates()

gnr_sds = {}
gnr_mns = {}
for v in vrbls:
    gnr_sds[v] = []
    gnr_mns[v] = []
    
for tgx in unq_tags[0:100]:

    dfcx = get_df_cbmd(tgx)

    for v in vrbls: 
        mnx, sdx = weighted_avg_and_std(dfcx[v], dfcx['cnt'])
        gnr_sds[v].append(sdx)
        gnr_mns[v].append(mnx)


plt.scatter(gnr_mns[v], gnr_sds[v])
plt.show()



# quite some parabolical relationships
# but definitely a lot of range


# cov mats are not that expensive to calculate at start me thinks
# also super cheap to store and access (dict)


# * scrap
# ** figuring out CH queries

# SELECT lfm_id as mbid, gender
# FROM acstb
# INNER JOIN
# (
#     SELECT mbid FROM (
#         SELECT * FROM (    
#             SELECT * FROM tag_sums 
#             JOIN 
#             (
#                 SELECT tag, count(*) AS tag_ap
#                 FROM tag_sums 
#                 WHERE (weight > 10) AND (rel_weight > 0.1)
#                 GROUP BY tag
#                 HAVING tag_ap > 5
#             ) USING (tag)
#         )
#         JOIN
#         (
#             SELECT mbid, cnt
#             FROM song_info3
#             WHERE cnt > 400
#         ) USING mbid
#         WHERE weight > 10 AND rel_weight > 0.1 
#     ) GROUP BY mbid
# ) USING mbid

# client_str = """select mbid, tag, rel_weight, cnt from 
# (select * from 
#     (select * from tag_sums
#         join (select tag, count(*) as tag_ap from tag_sums where weight > """ + str(min_weight) + """ and rel_weight > """ + str(min_rel_weight) + """
#             group by tag having tag_ap > """ + str(min_weight) + """) using tag)
#         join (select lfm_id as mbid from acstb) using mbid)
#     join (select mbid, cnt from song_info3 where cnt > """ + str(min_cnt) + """ )
#     using mbid where weight > """ + str(min_weight) + """ and rel_weight > """ + str(min_rel_weight)

    # client_str = "select lfm_id, " + vrbls_str + """ from acstb join 
    # (select mbid as lfm_id, cnt from song_info3 where cnt > """ + str(min_cnt) + ") using lfm_id"

    # rows_acst = client.execute(client_str)

    # client_str = "select lfm_id, " + vrbls_str + """ from acstb 
    # join ( select tag,count(*) as tag_ap from tag_sums where weight > 10)"""
    # rows_acst = client.execute("""select lfm_id, dncblt,gender,timb_brt,tonal,voice from acstb 
    # join (select mbid as lfm_id, cnt from song_info3 where cnt > 400 ) using lfm_id""")

    
# rows_tags = client.execute("""select mbid, tag, rel_weight, cnt from
# (select * from 
#     (select * from tag_sums
#         join (select tag, count(*) as tag_ap from tag_sums where weight > 10 and rel_weight > 0.05
#             group by tag having tag_ap > 10) using tag)
#     join (select lfm_id as mbid from acstb) using mbid)
# join (select mbid, cnt from song_info3 where cnt > 400 ) using mbid  
# where weight > 10 and rel_weight > 0.05""")


# popping entries no longer needed

# df_acst_dict = {}
# c = 0
# for r in df_acst.itertuples():
#     df_acst_dict[r.lfm_id] = c
#     c +=1

# # rows to pop
# rstp = []
# for i in songs_tbp:
#     # shouldn't every songs_tbp in df_acst???
#     # yup: songs_tbd are those that are in df_acst but not in df_tags
#     # which means they are in df_acst
#     rtp = df_acst_dict[i]
#     rstp.append(rtp)

# df_acst2 = df_acst.drop(rstp)
# # finally works

# ** reworked query for acst df

    # client_str = """
    # SELECT lfm_id as mbid,""" +  vrbl_strs + """
    # FROM acstb
    # INNER JOIN
    # (
    #     SELECT mbid FROM (
    #         SELECT * FROM (    
    #             SELECT * FROM tag_sums 
    #             JOIN 
    #             (
    #                 SELECT tag, count(*) AS tag_ap
    #                 FROM tag_sums 
    #                 WHERE (weight > """ + str(min_weight) + """) 
    #                 AND (rel_weight > """ + str(min_rel_weight) + """ )
    #                 GROUP BY tag
    #                 HAVING tag_ap > """ + str(min_tag_aprnc) + """
    #             ) USING (tag)
    #         )
    #         JOIN
    #         (
    #             SELECT mbid, cnt
    #             FROM song_info3
    #             WHERE cnt > """ + str(min_cnt) + """
    #         ) USING mbid
    #        WHERE weight > """ + str(min_weight) + """
    #        AND rel_weight > """ + str(min_rel_weight) + """

    #     ) GROUP BY mbid
    # ) USING mbid"""

    # ** rework2: use temporary tables
    # get tags that fulfill tag requirements in intersection

    # tag_sel = """
    # SELECT tag, count(tag) as tag_aprnc FROM (
    #     SELECT mbid, tag, weight, rel_weight FROM tag_sums
    #     JOIN
    #     ( """ + tag_song_sel + """
    #     ) USING mbid

    #     WHERE (weight > """ + str(min_weight) + """) 
    #     AND (rel_weight > """ + str(min_rel_weight) + """ )
    #     ) GROUP BY tag 
    #     HAVING tag_aprnc > """ + str(min_tag_aprnc)


    # # select combination
    # con_sel = """
    # SELECT * FROM (""" + tag_song_sel + """)
    # JOIN
    # (
    #     SELECT mbid, tag, weight, rel_weight FROM tag_sums
    #     JOIN 
    #     ( """ + tag_sel + """ )
    #     USING tag 
    #     WHERE (weight > """ + str(min_weight) + """) 
    #     AND (rel_weight > """ + str(min_rel_weight) + """ )
    # ) USING mbid
    # """

        # seems to work: all of the 530 have a at least 50 songs also in the df
    # need to combine with acoustic data now

    # JOIN (    
    #         SELECT mbid, tag, weight, rel_weight FROM tag_sums
    #             WHERE (weight > """ + str(min_weight) + """) 
    #             AND (rel_weight > """ + str(min_rel_weight) + """ )
    #     ) USING tag"""
    
    # JOIN (
    #     SELECT mbid, tag, weight, rel_weight FROM tag_sums
    #     JOIN
    #     ( """ + tag_song_sel + """
    #     WHERE (weight > """ + str(min_weight) + """) 
    #     AND (rel_weight > """ + str(min_rel_weight) + """ )
    #     ) USING mbid
    # ) USING tag
    # WHERE tag_aprnc > """ + str(min_tag_aprnc) + """ 
    # AND (weight > """ + str(min_weight) + """) 
    # AND (rel_weight > """ + str(min_rel_weight) + """ )
    # """

    # FUCKCCKCKCK
    # i never can get everything clear
    # if i get the tags of the songs which have at least one proper tag (weight + on proper genre), there will be tags for them that don't fulfill those criteria
    # maybe eventually unit of analysis is link between things (like song, tag and weight?)
    # but that's always unique, and no way of working on it



    # rows_acst = client.execute(client_str)
    # df_acst = pd.DataFrame(rows_acst, columns = ['lfm_id'] + vrbls)
    # songs_acst = df_acst['lfm_id']
    # print(len(np.unique(songs_acst)))

    # gets the mbids that can be used in terms of general requirements (min plcnt, acst, min tag requirement)
    # tag_song_sel = """
    # SELECT mbid from (""" + mbid_sel_fix + """)
    # JOIN
    # ( """ + mbid_sel_tag + """

    # ) USING mbid """



    
    
    # client_str2 = """
    # SELECT * FROM (
    #     SELECT mbid,tag, weight, rel_weight FROM tag_sums 
    #         JOIN
    #         (
    #             SELECT tag, count(tag) AS tag_aprnc FROM (
    #                 SELECT * FROM (
    #                     SELECT * FROM tag_sums 
    #                     INNER JOIN 
    #                     (
    #                         SELECT mbid, cnt
    #                         FROM song_info3
    #                         WHERE cnt > """ + str(min_cnt) + """
    #                     ) USING mbid
    #                     WHERE (weight > """ + str(min_weight) + """) 
    #                     AND (rel_weight > """ + str(min_rel_weight) + """ )
    #                 )
    #                 JOIN 
    #                 (
    #                     SELECT lfm_id AS mbid FROM acstb
    #                 ) USING mbid 
    #             )
    #             GROUP BY tag
    #             HAVING tag_aprnc > """ + str(min_tag_aprnc) + """
    #         ) USING tag
    #         WHERE (weight > """ + str(min_weight) + """) 
    #         AND (rel_weight > """ + str(min_rel_weight) + """ ))
    #     JOIN 
    #     (
    #         SELECT lfm_id as mbid from acstb)
    #     USING mbid"""

        # fuck this is so stupid
    # can't manage to combine both conditions, don't think properly now for 1 either
    # think final operation has to be join
    # connecting link is mbids
    # 1: get the musical info for a bunch of mbids
    # 2: get the tags for a bunch of mbids

        # mbid_sel_fix = """
    # SELECT lfm_id as mbid from acstb 
    # JOIN
    # (
    #     SELECT mbid from song_info3 
    #     WHERE cnt > """ + str(min_cnt) + """
    # ) USING mbid"""

        # generate string for musicological data
    
    # first join everything together, then filter on that
    # merging first: only counts those tags that are part of the genres that fulfil criteria
    # nope that's always teh case; some weightings (those of the tags I don't want) always get discarded

    # i hate all this nesting

    # client_str= """
    # SELECT lfm_id as mbid,""" + vrbl_strs + """
    # FROM acstb
    # ANY JOIN
    # (
    #     SELECT mbid,tag FROM tag_sums 
    #     JOIN
    #     (
    #         SELECT tag, count(tag) AS tag_aprnc FROM (
    #             SELECT * FROM (
    #                 SELECT * FROM tag_sums 
    #                 INNER JOIN 
    #                 (
    #                     SELECT mbid, cnt
    #                     FROM song_info3
    #                     WHERE cnt > """ + str(min_cnt) + """
    #                 ) USING mbid
    #                 WHERE (weight > """ + str(min_weight) + """) 
    #                 AND (rel_weight > """ + str(min_rel_weight) + """ )
    #             )
    #             JOIN 
    #             (
    #                 SELECT lfm_id AS mbid FROM acstb
    #             ) USING mbid 
    #         )
    #         GROUP BY tag
    #         HAVING tag_aprnc > """ + str(min_tag_aprnc) + """
    #     ) USING tag 
        
    # ) USING mbid

    # """

    # client.execute(mbid_tbl_basic)
    # client.execute(mbid_basic_insert)
    # client.execute('select count(*) from mbids_basic')

    # select songs that have at least 1 loading on greater than min_weight/min_rel_weight
    # on a genre with at least min_tag_aprnc other songs who do so as well

    # mbid_sel_tag = """SELECT distinct(mbid) from tag_sums 
    #     JOIN
    #     (    
    #         SELECT tag, count(tag) as tag_aprnc
    #         FROM tag_sums 
    #         WHERE (weight > """ + str(min_weight) + """) 
    #         AND (rel_weight > """ + str(min_rel_weight) + """ )
    #         GROUP BY tag
    #         HAVING tag_aprnc > """ + str(min_tag_aprnc) + """ 
    #     ) USING tag"""

