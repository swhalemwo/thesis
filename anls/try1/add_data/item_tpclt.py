import csv
import json
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

    # generate string for musicological data
    client_str = """
    SELECT lfm_id as mbid,""" +  vrbl_strs + """
    FROM acstb
    INNER JOIN
    (
        SELECT mbid FROM (
            SELECT * FROM (    
                SELECT * FROM tag_sums 
                JOIN 
                (
                    SELECT tag, count(*) AS tag_ap
                    FROM tag_sums 
                    WHERE (weight > """ + str(min_weight) + """) 
                    AND (rel_weight > """ + str(min_rel_weight) + """ )
                    GROUP BY tag
                    HAVING tag_ap > """ + str(min_tag_aprnc) + """
                ) USING (tag)
            )
            JOIN
            (
                SELECT mbid, cnt
                FROM song_info3
                WHERE cnt > """ + str(min_cnt) + """
            ) USING mbid
            WHERE weight > """ + str(min_weight) + """
            AND rel_weight > """ + str(min_rel_weight) + """
        ) GROUP BY mbid
    ) USING mbid"""

    rows_acst = client.execute(client_str)
    df_acst = pd.DataFrame(rows_acst, columns = ['lfm_id'] + vrbls)
    songs_acst = df_acst['lfm_id']
    
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
    # songs_tags = df_tags['lfm_id']
    # len(np.unique(songs_tags))

    return(df_acst, df_tags)

min_cnt = 200
min_weight = 10
min_rel_weight = 0.1
min_tag_aprnc = 20
min_cnt = 200


df_ac, df_tg = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc)
# seems to be working







    



    # make it into one function, they are mutually conditioned anyways


# rows_tags = client.execute("""select mbid, tag, rel_weight, cnt from
# (select * from 
#     (select * from tag_sums
#         join (select tag, count(*) as tag_ap from tag_sums where weight > 10 and rel_weight > 0.05
#             group by tag having tag_ap > 10) using tag)
#     join (select lfm_id as mbid from acstb) using mbid)
# join (select mbid, cnt from song_info3 where cnt > 400 ) using mbid  
# where weight > 10 and rel_weight > 0.05""")




songs_tbp = list(set(songs_acst) - set(songs_tags))

df_acst_dict = {}
c = 0
for r in df_acst.itertuples():
    df_acst_dict[r.lfm_id] = c
    c +=1

# rows to pop
rstp = []
for i in songs_tbp:
    # shouldn't every songs_tbp in df_acst???
    # yup: songs_tbd are those that are in df_acst but not in df_tags
    # which means they are in df_acst
    rtp = df_acst_dict[i]
    rstp.append(rtp)

df_acst2 = df_acst.drop(rstp)
# finally works

# make position dicts because dicts good

gnr_song_dict = {}
for r in df_tags.itertuples():
    gnr = r.tag
    
    if gnr in gnr_song_dict.keys():
        gnr_song_dict[gnr].append(r.lfm_id)
    else:
        gnr_song_dict[gnr] = [r.lfm_id]
        

acst_pos_dict = {}
for r in df_acst2.itertuples():
    acst_pos_dict[r.lfm_id] = r.Index


unq_tags = list(np.unique(df_tags['tag']))

gnr = 'rap'
gnr_acst_ids = [acst_pos_dict[i] for i in gnr_song_dict[gnr]]
df_gnr_tags = df_tags[df_tags['tag']==gnr]
df_gnr_acst = df_acst.loc[gnr_acst_ids]

df_gnr_cbmd = pd.merge(df_gnr_tags, df_gnr_acst, on='lfm_id')

trackx = '047a94b1-c42e-4867-8aa5-36d8c1f57d00'
trackx_loc = list(df_gnr_cbmd['lfm_id']).index(trackx)

vrbls = ['dncblt','gender', 'timb_brt','tonal', 'voice']



# still not clear about how to weigh, whether to use weight or count arrrrr
# hmm original idea was to use cnts and use batches based on weights

slx, ovrl = [], []


for i in vrbls:
    print(i)
    slx.append(float(df_gnr_cbmd[df_gnr_cbmd['lfm_id'] == trackx][i]))
    ovrl.append(float(weighted_avg_and_std(df_gnr_cbmd[i], df_gnr_cbmd['cnt'])[0]))


mat = df_gnr_cbmd[vrbls]

aweits = df_gnr_cbmd['cnt']

cov_mat = np.cov(mat, rowvar=0, aweights=aweits)
cov_mat_inv = np.linalg.inv(cov_mat)


from scipy.spatial.distance import mahalanobis
mahalanobis(slx, ovrl, cov_mat_inv)





# have to scale it somewhow: the variance matters in the assessment of typicality
# average z score?
# multidimensional z score? -> Mahalanobis distance

## * scrap
## ** figuring out CH queries

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
