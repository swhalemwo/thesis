# from clickhouse_driver import Client
import csv
import string
import random
from collections import Counter
import json
import urllib.request
import requests
import time
from clickhouse_driver import Client
client = Client(host='localhost', password='anudora', database='frrl')

# import musicbrainzngs

# top_mbids=client.execute('select mbid from song_info3 order by cnt desc limit 20')

##########################
# retrieving information #
##########################

# hm how should i handle mibd insufifficiencies (mlhd id differeing from mbid)
# get mb mbid first?
# i think MB api is kinda fast? also think it has batch requests?
# nope, MB api is slow af (1 request/sec)
# had to rewrite tag retrieval script: split, also store mbid



# i = some_mbids[9]

    
# batch processing testing

def batch_prepper(batch):
    """preps url for api from batch"""
    base_str = 'https://acousticbrainz.org/api/v1/high-level?recording_ids='
    # base_str = 'https://acousticbrainz.org/api/v1/low-level?recording_ids='
    cntr = 0
    batch_str = ""
    for i in batch:
        cntr +=1
        batch_str = batch_str + i
        if cntr < len(batch):
            batch_str = batch_str + ';'

    url = base_str + batch_str
    return(url)


# maximum torture, but seems to be working?
# even with low -level still only needs ~2 sec for 31
# much faster with high level

def batch_procr(data2, mlhd_ids, pointers):
    """loop over input list, process based on what wored (lfmid, mbid) """
    for i in mlhd_ids:

        # first easy case: i not in pointers
        if i not in pointers.keys():

            if i in data2.keys():
                # print('gotcha')
                skes_proc(i, None)

            else:
                fail_proc(i)
                # print('fail')

        # more difficult cases: uneqs -> 4 cases: both, neither, 01, 10
        # i = list(pointers.keys())[2]
        # if i in pointers.keys():
        else:
            v = pointers[i]

            i_stat = i in data2.keys()
            v_stat = v in data2.keys()

            if i_stat == True and v_stat == True: skes_proc(i, None)
            if i_stat == True and v_stat == False: skes_proc(i, None)
            if i_stat == False and v_stat == True:
                print(i, v)
                indirects.append(i)
                skes_proc(i,v)
            if i_stat == False and v_stat == False: fail_proc(v)


def skes_proc(j, v):
    """If j in mlhd_ids and datat2; dict used depends on which dict works"""
    
    if v is not None:
        # add all the other data here
        # svl = data2[v]['0']['highlevel']['danceability']['all']['danceable']
        mus_dt = data_proc(data2[v]['0']['highlevel'])
        meta_dt = metadata_proc(data2[v]['0']['metadata'])

    else:
        # svl = data2[j]['0']['highlevel']['danceability']['all']['danceable']
        mus_dt = data_proc(data2[j]['0']['highlevel'])
        meta_dt = metadata_proc(data2[j]['0']['metadata'])

    skes.append([j] + mus_dt + meta_dt)

def fail_proc(j):
    fails.append([j])

def writer_res(skes, fails):
    with open(ACST_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(skes)
        
    with open(FAIL_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(fails)
        

# are they measuring distinct things? 

def data_proc(inf_dict):
    """gets musicological information out of information dict, puts in inf_row list"""
    dncblt = inf_dict['danceability']['all']['danceable']
    gender = inf_dict['gender']['all']['female']
    timb_brt = inf_dict['timbre']['all']['bright']
    tonal = inf_dict['tonal_atonal']['all']['tonal']
    voice = inf_dict['voice_instrumental']['all']['voice']

    moods = [inf_dict["mood_" + i]['all'][i] for i in mood_keys]

    mus_inf = [dncblt, gender, timb_brt, tonal, voice] + moods
    

    gnrs_drtmnd = [inf_dict['genre_dortmund']['all'][i] for i in gnr_drtmnd_keys]
    gnrs_rosemern = [inf_dict['genre_rosamerica']['all'][i] for i in gnr_rosmern_keys]
    gnrs_tzan = [inf_dict['genre_tzanetakis']['all'][i] for i in gnr_tzan_keys]
    moods_mirex = [inf_dict['moods_mirex']['all'][i] for i in moods_mirex_keys]

    gnr_inf = gnrs_drtmnd + gnrs_rosemern + gnrs_tzan + moods_mirex

    inf_row = mus_inf + gnr_inf
    return(inf_row)



def nested_get(input_dict, nested_key, deflt):
    """general fun from SO to get nested entries from dicts"""
    internal_dict_value = input_dict
    for k in nested_key:
        internal_dict_value = internal_dict_value.get(k, None)
        if internal_dict_value is None:
            return deflt
    return internal_dict_value


def metadata_proc(md_dict):
    """gets some metadata"""
    lang = nested_get(md_dict, ['tags', 'language'], ['NOLANG'])[0]
    # print(len(nested_get(md_dict, ['tags', 'language'], ['NOLANG'])))
    
    length = nested_get(md_dict, ['audio_properties','length'], -1)
    label = nested_get(md_dict, ['tags','label'], ["NOLABEL"])[0]
    # print(len(nested_get(md_dict, ['tags','label'], ["NOLABEL"])))
    
    rl_type = nested_get(md_dict, ['tags','release type'], ["NORLTYPE"])[0]
    rls_cri = nested_get(md_dict, ['tags','musicbrainz album release country'], ["NORLSCRI"])[0]
    # print(len(nested_get(md_dict, ['tags','musicbrainz album release country'], ["NORLSCRI"])))

    md_row = [length, label, lang, rl_type, rls_cri]
    return(md_row)


def get_dones():
    with open(ACST_FILE) as fi:
        rdr = csv.reader(fi)
        skes = [r[0] for r in rdr]

    with open(FAIL_FILE) as fi:
        rdr = csv.reader(fi)
        failed = [r[0] for r in rdr]

        dones = skes + failed
        return(dones)
        
# md_dict = data2['fb47ca87-499e-4c49-b8a1-3f784d1daa1b']['0']['metadata']


# i = 'fb47ca87-499e-4c49-b8a1-3f784d1daa1b'
# inf_dict = data2['fb47ca87-499e-4c49-b8a1-3f784d1daa1b']['0']['highlevel']


# for i in list(data2.keys()):
#     md_dict = data2[i]['0']['metadata']
#     x = metadata_proc(md_dict)

# define general keys genres and moods
gnr_drtmnd_keys = ['alternative', 'blues', 'electronic', 'folkcountry', 'funksoulrnb', 'jazz', 'pop', 'raphiphop', 'rock']
gnr_rosmern_keys = ['cla', 'dan', 'hip', 'jaz', 'pop', 'rhy', 'roc', 'spe']
gnr_tzan_keys = ['blu', 'cla', 'cou', 'dis', 'hip', 'jaz', 'met', 'pop', 'reg', 'roc']
moods_mirex_keys = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5']
mood_keys = ['acoustic', 'aggressive', 'electronic', 'happy', 'party', 'relaxed', 'sad']


if __name__ == '__main__':

    # with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split2/1_addgs.csv', 'r') as fi:
    #     rdr = csv.reader(fi)
    #     some_mbids = [i[0:2] for i in rdr]

    some_mbids = client.execute("""select lfm_id, mbid from addgs join 
                                (select lfm_id, count(lfm_id) as cnt from addgs group by lfm_id having cnt =1) 
                                using (lfm_id)""")

    ACST_FILE = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/acstbrnz.csv'
    FAIL_FILE = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/acst_fails.csv'

    dones = get_dones()

    smids = [i[0] for i in some_mbids]

    stpd_dict = {}
    for i in some_mbids:
        stpd_dict[i[0]] = i[1]
    
    for i in dones:
        try:
            x = stpd_dict.pop(i)
        except:
            pass

    # some_mbids2 = [[k,v]

    some_mbids2 = []
    for k in list(stpd_dict.keys()):
        some_mbids2.append([k, stpd_dict[k]])

    some_mbids = some_mbids2

    batch = []
    mlhd_ids = []
    pointers = {}

    indirects=[]
    skes = []
    fails =[]

    

    for i in some_mbids:
        if i[0] == i[1]:
            batch.append(i[0])
            mlhd_ids.append(i[0])
        else:
            batch.append(i[0])
            batch.append(i[1])

            mlhd_ids.append(i[0])

            pointers[i[0]] = i[1]
            # pointers[i[1]] = i[0]

        if len(batch) > 40:
            print('process batch')
            print(some_mbids.index(i))

            # break
            batch_str=batch_prepper(batch)
            # t1 = time.time()
            # for p in range(200):
            with urllib.request.urlopen(batch_str) as url2:
                data2 = json.loads(url2.read().decode())

            batch_procr(data2, mlhd_ids, pointers)
            writer_res(skes, fails)

            batch = []
            mlhd_ids = []
            pointers = {}

            indirects=[]
            skes = []
            fails =[]

        

    #     print(len(data2))
    # t2 = time.time()


###################################
# make colnames for R or whatever #
###################################
    
colx1 = ['dncblt', 'gender', 'timb_brt', 'tonal', 'voice'] +  ['mood_' + i for i in mood_keys]

col_names1 = ['gnr_dm_' + i for i in gnr_drtmnd_keys]
col_names2 = ['gnr_rm_' + i for i in gnr_rosmern_keys]
col_names3 = ['gnr_tza_' + i for i in gnr_tzan_keys]
col_names4 = ['mirex_' + i for i in moods_mirex_keys]

colx2 = col_names1 + col_names2 + col_names3 + col_names4

colx3 = ['length', 'label', 'lang', 'rl_type', 'rls_cri']

colnames = ['id'] + colx1 + colx2 + colx3



# hmm not sure if i should use offiical genres
# do they add something?
# are so general -> will not die out: will be not that many (maybe some hundred)
# would have to be split up with quite some work
# if lfm genre dies out, look up what happens here? -> see how much agreement there was in terms of official genres?
# should really see if i can get official genre information from MB: more complete: nope not available?? at least not through api, 
# don't know where the 
# would be nice to get interaction between informal and "official" classification systesm
# also way of control: do users just parrot officials?
# but MBID genre information is also user-provided
# discogs?
# allmusic seems pretty good (https://labs.acousticbrainz.org/dlfm2016/)
# based on tivo, is on album level?
# there is something on song level, but not that much apparently?
# could also use moods/themes
# but in either case it's separately from acst_brainz





    # with open(ACST_FILE, 'a') as fo:
    #     wr = csv.writer(fo)
    #     wr.writerow(prntrow)


# data[i]['0']['metadata']['audio_properties']['length']
# data[i]['0']['highlevel'].keys()
# data[i]['0']['highlevel']['timbre']['all']['bright']



####### high level stuff
# https://acousticbrainz.org/datasets/accuracy#genre_dortmund
# danceability: 0-1 (probability same)
# gender: female 0-1, male 0-1 (probability is of higher entry)
# genre_dortmund: amount of genres (alternative, blues, electronic, folk-country, funksoulrnb, jazz, etc) -> pointless
# genre_electronic: amount of 5 electronic genres
# genre_rosamerica: other genre classification
# genre_tzanetakis: other genre classification
# ismir04_rhythm: dance style
# mood_acoustic: 0-1
# mood_aggressive: 0-1
# mood_electronic: 0-1
# mood_happy: 0-1
# mood_party: 0-1
# mood_relaxed: 0-1
# mood_sad: 0-1
# moods_mirex: 5 clusters (passionate, cheerful, literate, humerous, aggressive)
# timbre:  0-1
# tonal_atonal: 0-1
# voice_instrumental: 0-1


# useful:
# - danceability
# - gender
# - moods
# - timbre
# - tonal_atonal
# - voice_instrumental

# groups for genre spanning (which songs span genres)? 
# - maybe moods_mirex: just save them: might also be useful for genre-spanning?
 

# Piazzai:
# length,
# danceability
# main key
# scale
# frequency of main key
# chord progression
# scale of chord key
# bpm
# total count of beats

# could make separate tables with song - album - artist
# if i want to use some more industry-level explanatory mechanisms

# -> uses high level data, but IS SUBJECT TO RECALCULATION




##################
# MB API testing #
##################

# WERKS
# https://musicbrainz.org/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4?inc=artists+releases
# # not sure if faster than python api tho
# # NOPE 


# https://musicbrainz.org/ws/2/url/ws/2/recording/4843d67e-e3e3-47d0-813b-d4d9f0cb6d56?inc=artists
# https://musicbrainz.org/ws/2/url/ws/2/recording/0f061025-a50e-44d5-8853-86d9ae3d09b9?inc=artists

# https://musicbrainz.org/ws/2/url/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4
# https://musicbrainz.org/ws/2/url/ws/2/recording/96685213-a25c-4678-9a13-abd9ec81cf35



# TIME TESTING

# t1 = time.time()
# for i in range(2000):
#     url = 'https://musicbrainz.org/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4?inc=artists+releases&fmt=json'
#     resp_raw = requests.get(url)
#     resp_dict = resp_raw.json()
#     print(len(resp_dict))
# t2 = time.time()

# API starts blocking requests when they're too close to each other

# t1 = time.time()
# for i in range(20):
#     mb_inf = musicbrainzngs.get_recording_by_id('c69310f9-e2e5-4fb2-ac26-836913c478d4', includes=['releases', 'artists'])
#     print(i)
    
# t2 = time.time()

# python api adheres to 1 request/second rule

###############################################
# trying to get genres from MBU, doesn't work #
###############################################

# mb_inf = musicbrainzngs.get_recording_by_id(i, includes=['releases', 'artists'])
# mb_inf = musicbrainzngs.get_recording_by_id(i, includes=['genres'])

# musicbrainzngs.get_release_group_by_id

# for i in some_mbids[0:10]:
#     url = "https://musicbrainz.org/ws/2/recording/" + i[1] + "?inc=genres&fmt=json"
#     try:
#         resp_raw = requests.get(url)
#         resp_dict = resp_raw.json()
#         print('works')
#         try:
#             print(resp_dict['genres'])
#         except:
#             pass
#     except:
#         print('fail')
#         pass

#     time.sleep(0.9)

# https://musicbrainz.org/ws/2/release/517ed123-f71d-4320-b27e-d235fec80dcd?inc=genres


# https://musicbrainz.org/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4?inc=genres

# https://musicbrainz.org/ws/2/recording/2ee68ec3-d85b-4bc9-8f65-3a109af26b5a?inc=user-genres

# https://musicbrainz.org/ws/2/recording/778850f8-b9b9-475b-900f-1c0114ca729f?inc=genres+artists

# for i in list(data2.keys()):
#     try:
#         print(data2[i]['0']['metadata']['tags']['genre'])
#     except:
#         print('lol')
#         pass
#         print(i)



####################### figuring out API

# calling with 9: return5, seemingly random order (1,3,4,6,9)
# could just be those that work?
# yup
# seems possible to get input of 48 max
# yup: 49 breaks it even when all valid -> batches of 48

# that poor api omg

# if i[0] != i[1]: send both

# need way to manage mlhd ids



################################################
# testing if things work, not that big concern #
################################################
    
# for i in indirects:
#     # print(i in list(data2.keys()))
#     # print(pointers[i] in list(data2.keys()))
    
#     # print(i in [k[0] for k in skes])
#     print(pointers[i] in [k[0] for k in skes])
    

# NEED TO WRITE EXCEPTION WHEN YOU HAVE 2 different mlhd ids point to the same mbid (which is different from both)
# currently would overwrite dicts

# probably same in other direction too?
# NOPE: all items in first row unique

# are reverse pointers ever needed?
# maybe not because i'm looping over mlhd ids which are unique?
# i think actually not: not asked ever anyways? 
# final attribution is to original unique mlhd_id anyways


###########################
# old method for metadata #
###########################

    # length = -1
    # label = 'NOTTHERE'
    # lang = 'NOLANG'
    # rl_type = "NORLTYPE"
    # rls_cri = "NORLSCRI"

    # length = md_dict['audio_properties']['length']
    # label = md_dict['tags']['label'][0]
    # lang = md_dict['tags']['language']
    # rl_type = md_dict['tags']['release type']
    # rls_cri = md_dict['tags']['musicbrainz album release country']

        
