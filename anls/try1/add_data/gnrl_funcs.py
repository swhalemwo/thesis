def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))



def get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
            min_unq_artsts, max_propx1, max_propx2, d1, d2, ptn,
            usr_dedcgs, tag_plcnt, unq_usrs,
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
    ptn: partition in usrsNk/ptn
    usr_dedcgs: usr dedication: how many unique songs of genre usr needs to listen to be counted as listener
    tag_plcnt: how large playcount of genre has to be for user to be counted as listener
    unq_usrs: how many dedicated users genre has to have to be considered

    """

    # ptn = 1

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
    cnt Float32,
    artist String,
    erl_rls Int32,
    len_rls_lst Int8
    )
    """
    # d1 = '2011-10-01'
    # d2 = '2011-11-01'
    
    # ptn = 1

    # filters by date and usrs having the right partition
    # date_str = """SELECT mbid, cnt FROM (
    #     SELECT * FROM (
    #         SELECT song as abbrv, count(song) as cnt FROM (
    #             SELECT usr, song from logs
    #                 WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'
    #             ) JOIN (SELECT abbrv2 AS usr FROM usrs4k WHERE ptn == """ + str(ptn) + """
    #             ) USING usr
    #             GROUP BY song
    #             HAVING cnt > """ + str(min_cnt) + """
    #     ) JOIN (
    #         SELECT mbid, abbrv FROM song_info) 
    #         USING abbrv
    #     )"""


    # SELECT usr, song, cnt, mbrshp, cnt*mbrshp as cnt_mbrshp  FROM (
    date_str = """
    SELECT mbid, cnt_mbrshp as cnt FROM (
        SELECT song as abbrv, SUM(cnt*mbrshp) as cnt_mbrshp  FROM (
            SELECT usr, song, count(song) as cnt FROM (
                SELECT usr, song from logs  WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """')
                JOIN (SELECT usr from ptn) USING usr
            GROUP BY (usr, song)
        ) JOIN (SELECT usr, ptn"""+ str(ptn) + """ as mbrshp from ptn
        ) USING usr
        GROUP BY song
        HAVING cnt_mbrshp > """ + str(min_cnt) + """
    ) JOIN (
        SELECT mbid, abbrv FROM song_info) 
        USING abbrv
        """

# now i've partioned users and songs, but i don't use the coefficients of songs
# but what about the songs that are not used for partioning
# didn't have that problem in networks because of one-mode conversion, but now i have data
# i could limit myself to the songs that i use to partition the usrs, but would mean dropping many
# acoustic info is cheap, so no reason to throw stuff away


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
    cnt Int32,
    tag String,
    weight Int8,
    rel_weight Float32, 
    artist String, 
    erl_rls Int32, 
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
        SELECT mbid_basic as mbid, cnt, artist, erl_rls, len_rls_lst FROM mbids_basic)
    USING mbid"""
    
    # get tags that are actually present enough in intsec
    # use actual table since used again to get users
    

    
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
    

    int_sect_all_qry = """
    CREATE TEMPORARY TABLE int_sect_all (
    mbid String, 
    cnt Int32,
    tag String,
    weight Int8,
    rel_weight Float32,
    artist String,
    erl_rls Int32,
    len_rls_lst Int8)
    """

    # boil down basic_songs_tags to intersection requirements
    int_sect_all = """
    INSERT INTO int_sect_all
    SELECT * from basic_songs_tags
    JOIN ( """ + intsect_tags + """)
    USING tag"""
    

    # make merge table by getting stuff from acstb in
    # filtered on acstb before so should all be in there, and seems like it is

    
    # try to work in usrs count
    # usr_dedcgs: user dedication: how many unique songs a user needs to have to be counted as listening to it
    # unq_usrs: how many users a genre needs to be considered
    # maybe add something like playcount? 
    # is a bit tempting to add all the user information tbh
    # maybe later; this is about genre filtering
    
    # my listening counts for songs are now still taking into account listening events i delete later on
    # not good
    # i think it mostly affects small genres
    # compare results of usr_fltr with size_dict_raw

    # but usr gnr playcount not weighted 
    # i'm not sure if it should be: it's really more a minimum threshold, not an indicator of playcount
    # but that means it's not possible to compare how much my song playcount estimates are affected by NOT EXCLUDING??
    # listening events of marginally interested people are now included in popular genres
    # although they only listen once or twice these events are included because others listen a whole lot more
    # can't see how that's supposed to have a large impact on actual genres, the main impact is on small ones which get rightly excluded
    
    usr_fltrd_tags_qry = """
    CREATE TEMPORARY TABLE usr_fltrd_tags (
    tag String,
    unq_usrs Int32,
    sum_tag_plcnt Int32)
    """

    usr_fltr = """
    INSERT INTO usr_fltrd_tags
    SELECT tag, uniqExact(usr) as unq_usrs, sum(tag_plcnt) as sum_tag_plcnt FROM (
        SELECT tag, usr, uniqExact(abbrv) as usr_dedcgs, sum(cnt_abbrv) as tag_plcnt FROM 
            (SELECT usr, song as abbrv, count(abbrv) as cnt_abbrv from logs
                WHERE time_d BETWEEN '""" + d1 + """' and '""" + d2 + """'  
                GROUP BY (usr, song))
        JOIN (SELECT mbid, abbrv, tag FROM (
                SELECT mbid,tag FROM basic_songs_tags 
                    JOIN (
                        SELECT distinct(tag) from int_sect_all
                    ) using tag
             ) JOIN (SELECT mbid, abbrv from song_info) USING mbid
        ) USING abbrv

        GROUP BY (usr, tag)
        HAVING usr_dedcgs > """ + str(usr_dedcgs) + """ AND tag_plcnt > """ + str(tag_plcnt) + """
        
    ) GROUP BY tag
    HAVING unq_usrs > """ + str(unq_usrs)

    # harsh: 
    # usr_dedcgs: 4
    # tag_plcnt: 9
    # unq_usrs: 14


    int_sec_all2 = """
    SELECT * from int_sect_all JOIN (SELECT tag from usr_fltrd_tags) using tag
    """
    
    merge_qry = """
    SELECT lfm_id as mbid, cnt, tag, weight, rel_weight, artist, erl_rls, len_rls_lst, """ + vrbl_strs + """ from acstb2
    JOIN (""" + int_sec_all2 + """) USING mbid"""
    
    # probably should have a separate one to see group user by tag to make sure that user has at least 5 uniq songs on tag
    # can actually done nicely iteratively
    # question is how many uniqe songs does a user need to be counted to a tag: 3 sounds reasonable
    # how many users does tag need? lets say 15
    # relevant because it reduces number of genres
    # need integration anyways 
    # maybe intsec all should be a table, it's called more than once now
    
    drops = [
        'drop table mbids_basic',
        'drop table tags_basic',
        'drop table basic_songs_tags',
        'drop table int_sect_all',
        'drop table usr_fltrd_tags'
    ]
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
    client.execute(int_sect_all_qry)
    client.execute(int_sect_all)
    client.execute(usr_fltrd_tags_qry)
    client.execute(usr_fltr)
    
    rows_merged = client.execute(merge_qry)

    dfc = pd.DataFrame(rows_merged, columns = ['lfm_id','cnt', 'tag', 'weight', 'rel_weight',
                                               'artist', 'erl_rls', 'len_rls_lst'] + vrbls)
    # generate string for tag data
    return(dfc)

################## get_dfs test start ############

# min_cnt = 10
# min_weight = 10
# min_rel_weight = 0.1
# min_tag_aprnc = 30
# d1 = '2011-05-01'
# d2 = '2011-05-31'

# client = Client(host='localhost', password='anudora', database='frrl')

# # vrbls = ['dncblt','gender','timb_brt','tonal','voice']

# vrbls = ['dncblt','gender','timb_brt','tonal','voice','mood_acoustic','mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 


# dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc, d1, d2)
# gnrs = list(np.unique(dfc['tag']))

################## get_dfs test end ############

def dict_gnrgs(dfc, gnrs, pd):
    """generates dict with genres as keys as their cmbd dfs as values"""
    # make position dicts because dicts good
    # or rather, they would be if pandas subsetting wouldnt be slow AF
    # solution: make genre dfs only once in initialization

    acst_gnr_dict = {}
    for r in dfc.itertuples():
        gnr = r.tag

        if gnr in acst_gnr_dict.keys():
            acst_gnr_dict[gnr].append(r)
        else:
            acst_gnr_dict[gnr] = [r]

    for gnr in gnrs:
        acst_gnr_dict[gnr] = pd.DataFrame(acst_gnr_dict[gnr])

    return(acst_gnr_dict)

# acst_gnr_dict = dict_gnrgs(dfc)


# * general comparison function

# order doesn't matter anymore due to standardization
# do i need a matrix?
# can read it back in (either lower triangle columnwise or upper rowise)
# not sure if needed tho
# can just the positions of those over threshold, get the corresponding original comparisions, get the corresponding genres, and add that to edge list

# nested loops to ensure direction
# smaller one is now first: relation to be tested is subsetness
# but there is no a priori reason why smaller genre should be subset
# subgenre can also grow larger than original (w40k > warhammer)


def cmp_crubgs(gnrs, vd):

    gnr_ids = [vd[i] for i in gnrs]
    
    lenx = len(gnrs)

    cprx = []
    c = 0
    for i in range(lenx):

        cprx2 = []

        for k in range(i,lenx):

            # print(k)

            if i==k:
                next
            else:
                c +=1

                v1, v2 = gnrs[i], gnrs[k]
                v1_sz, v2_sz = sz_dict[v1], sz_dict[v2]

                if v1_sz > v2_sz:
                    cprsn = [vd[v2], vd[v1]]
                else:
                    cprsn = [vd[v1], vd[v2]]

                cprx2.append(cprsn)
        if len(cprx2) > 0:
            cprx.append(cprx2)

    cmps = list(itertools.chain.from_iterable(cprx))
    return(cmps)

# cmps = cmp_crubgs(gnrs, vd)


# cmps = all_cmps_crubgs(gnrs, vd, 'product')


