from  clickhouse_driver import Client
import numpy as np
import pandas as pd
from random import sample
from numpy.random import choice
import time
import itertools
import copy
from collections import Counter
import argparse

# from scipy.sparse.csr_matrix import sum as sprs_sum


from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from scipy.sparse import save_npz, load_npz
from joblib import dump, load


client = Client(host='localhost', password='anudora', database='frrl')

# min_song_plcnt = 30 # song has to be played at least these many times
# min_usr_cnt = 25 # song has to be listened to by at least that many users
# min_usr_plcnt = 50 # user has to play at least that many (unique) songs (which in turn have at least min_usr_cnt unique users playing them)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('d1', help='start of period')
    parser.add_argument('d2', help='end of period')
    parser.add_argument('min_song_plcnt', help='minimal total playcount of song')
    parser.add_argument('min_usr_cnt', help='number of minimum unique listeners')
    parser.add_argument('min_usr_plcnt', help='minimum number of unique songs per user')

    args = parser.parse_args()
    d1 = args.d1
    d2 = args.d2
    min_song_plcnt = int(args.min_song_plcnt)
    min_usr_cnt = int(args.min_usr_cnt)
    min_usr_plcnt = int(args.min_usr_plcnt)

    # d1='2010-08-28'
    # d2='2010-11-20'

    sel_str ="""INSERT INTO el 
        SELECT usr, song, count(usr,song) as cnt FROM (
            SELECT * FROM (
                SELECT usr, song, time_d FROM logs
                WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
            )

            JOIN (SELECT song, count(song) as song_cnt FROM logs
                WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
                GROUP BY song
                HAVING song_cnt > """ + str(min_song_plcnt) + """
                ) USING song

        ) JOIN (SELECT abbrv2 as usr from usr_info)
        USING usr
        GROUP BY (usr,song)"""

    # WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'

    try:
        client.execute('drop table el')
    except:
        pass

    print('creating temporary table')
    
    client.execute("CREATE TEMPORARY TABLE el (usr String, song String, cnt Int16)")
    client.execute(sel_str)
    client.execute("select count(*) from el")
    client.execute("select uniqExact(usr) from el")

    usr_string = """
    SELECT usr, song, cnt FROM (
        SELECT usr,song, cnt FROM el
        JOIN (
            SELECT song, count(song) as song_cnt2 FROM el
            GROUP BY song 
            HAVING song_cnt2 > """ + str(min_usr_cnt) + """
        ) USING song
    ) JOIN ( SELECT usr, count(usr) FROM (
        SELECT usr,song, cnt FROM el
            JOIN (SELECT song, count(song) as song_cnt2 FROM el
            GROUP BY song 
            HAVING song_cnt2 > """ + str(min_usr_cnt) + """
            )  USING song
        ) GROUP BY usr 
        HAVING count(usr) > """ + str(min_usr_plcnt) + """
    ) USING usr
    """

    print('getting usr links')
    usr_trk_lnks = client.execute(usr_string)

    unq_usrs = np.unique([i[0] for i in usr_trk_lnks])

    usr_song_dict = {}
    for u in unq_usrs: usr_song_dict[u] = {}

    for lnk in usr_trk_lnks:
        usr_song_dict[lnk[0]][lnk[1]]=lnk[2]

    usr_song_dicts = [usr_song_dict[i] for i in unq_usrs]

    v = DictVectorizer()
    X = v.fit_transform(usr_song_dicts)

    # usr_sums = X.sum(axis=1)
    song_sums = np.array(X.sum(axis=0))[0]

    rnd_sel = choice(range(X.shape[1]), 5000)
    X2_rnd = X[:,rnd_sel]
    usr_trk_lnks = 0

    t1= time.time()

    ldax = LatentDirichletAllocation(n_jobs = 3,
                                     n_components=5, 
                                     max_iter = 60,
                                     doc_topic_prior = 0.1,
                                     topic_word_prior = 0.4,
                                     verbose=3)
    ldax.fit(X2_rnd)
    scr = ldax.score(X2_rnd)

    mbrshp = ldax.transform(X2_rnd)
    t2= time.time()

    ptn_str = """CREATE table ptn (usr String, 
    """ + ', '.join(['ptn' + str(i) + ' Float32' for i in range(mbrshp.shape[1])]) + """,
     rndm Int8) ENGINE = MergeTree()
    PARTITION BY rndm
    ORDER BY tuple()"""

    try:
        client.execute('drop table ptn')
    except:
        pass

    client.execute(ptn_str)


    ptn_rows = []
    for u in zip(unq_usrs, mbrshp):
        ptn_row = [u[0]] + list(u[1]) + sample(range(10), 1)
        ptn_rows.append(ptn_row)

    client.execute('INSERT INTO ptn values', ptn_rows)

    save_mat = pd.DataFrame(mbrshp)
    save_mat['dt'] = d1 + " -- " + d2
    save_mat['usr'] = unq_usrs

    mat_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/mbrshp_mat/'
    
    save_mat.to_csv(mat_dir + d1 + " -- " + d2 + '.csv')





# * scrap
# ** seeing which whether random or weighted selection gets better indegree correlation
# wtd_sel = choice(range(X.shape[1]), 5000, p = song_sums/sum(song_sums))

# # X2_rnd = X[:,rnd_sel]
# X2_wtd = X[:,wtd_sel]

# x1 = np.concatenate((usr_sums, X2_rnd.sum(axis=1)), axis =1)
# x2 = np.concatenate((usr_sums, X2_wtd.sum(axis=1)), axis =1)

# np.corrcoef(x1.T)
# np.corrcoef(x2.T)
# random selection gives higher corcoef, better user degree
