import csv
import requests
import argparse
from clickhouse_driver import Client
import os
import re
import datetime
from datetime import timedelta
from calendar import monthrange
import pylast
import musicbrainzngs
musicbrainzngs.set_useragent('tagler', 0.1, 'lel')


# mbid = '94a2a5ba-390d-4dd9-9f69-3e314f8b9d98'
# mbid= '0e37764b-0726-4a15-8c16-abc07c4ea033'

# mbid = i
# mbid = failed[300]

# mbid2 = musicbrainzngs.get_recording_by_id(failed[300])['recording']['id']
# mbid = mbid2

# for i in dones[:50]:


# def get_lfm_fulls_from_mlhd(idx):
# # fulls: opposite to mbid
#     str1="http://ws.audioscrobbler.com/2.0/?method="
#     str2='track' + ".getInfo&mbid="    
#     str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

#     qry = str1 + str2 + idx + str4

#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()

#     return(resp_dict)


# rls_lst = mb_inf['recording']['release-list']
# date_prcsr(rls_lst)

def date_prcsr(rls_lst):
    """Takes list of releases, returns 
    - earliest date (9999-9-9 if no date)
    - level with highest resolution (0 if no date)
    - highest resolution of earliest date (-1 if no date)
    - if earliest date earlier than higher resolved date: time difference between earliest and highest resolved date (0 if no date or if highest resolved date is earliest)
    - length of release list
    - length of release list with dates
    """
    
    # add number of releases? 
    rlss_long=[]
    rlss_medm=[]
    rlss_shrt=[]

    len_rls_lst_ttl = len(rls_lst)
    
    for i in rls_lst:
        try:
            dt = i['date']

            if len(dt) == 10:
                dt2 = [int(i) for i in dt.split('-')]
                dttm = datetime.datetime(dt2[0], dt2[1], dt2[2])
                rlss_long.append(dttm)

            if len(dt) == 7:
                dt2 = [int(i) for i in dt.split('-')]
                max_days = monthrange(dt2[0], dt2[1])[1]
                
                dttm = datetime.datetime(dt2[0], dt2[1], max_days)
                rlss_medm.append(dttm)

            if len(dt) == 4:
                dttm = datetime.datetime(int(dt), 12, 31)
                rlss_shrt.append(dttm)
        except:
            pass

    lowest_lvl = 0
    if len(rlss_shrt) > 0: lowest_lvl=1
    if len(rlss_medm) > 0: lowest_lvl=2
    if len(rlss_long) > 0: lowest_lvl=3

    if lowest_lvl ==0:
        min_ttl = datetime.datetime(9999,9,9)
        max_mbrshp = -1
        len_rls_dts = 0
        tdiff = 0
        # not sure what best default value is: could be e.g. -1, 0, 99
    else:
        all_dts = rlss_long + rlss_medm + rlss_shrt
        min_ttl = min(all_dts)

        len_rls_dts = len(all_dts)

        buckets = [rlss_shrt, rlss_medm, rlss_long]

        min_ttl_mbrshps = []
        cntr = 1
        for i in buckets:
            if min_ttl in i:
                min_ttl_mbrshps.append(cntr)
            cntr +=1

        tdiff = 0 

        max_mbrshp = max(min_ttl_mbrshps)

        if max_mbrshp < lowest_lvl:
            tdiff = min_ttl - min(buckets[lowest_lvl-1])
            tdiff=tdiff.days

            print("DATE ISSUE")
            # maybe add some return thing: return both max(min_ttl_mbrshps) and lowest level
            
    return([min_ttl, lowest_lvl, max_mbrshp, tdiff, len_rls_dts, len_rls_lst_ttl])



def get_song_obj(arg_list, default = None):
    """More sophisticated calling of different ways to get song object"""
    orgn_dict = {1:'lfm', 2:'mb', 3:"manual"}
    fun_list = [network.get_track_by_mbid, network.get_track_by_mbid, network.get_track]
  
    cntr = 1
    for f in fun_list:

        argx = arg_list[cntr-1]
        try:
            sx = f(*argx)

            return([orgn_dict[cntr], sx])
        except:
            cntr +=1
            continue
    else:
        return 'song not found nowhere'
    
# i = failed[0]
# mb_inf = get_mb_inf(i)
# a = mb_inf['recording']['artist-credit'][0]['artist']['name']
# t = mb_inf['recording']['title']
# arg_list = [[i], [idx], [a, t]]
# songx=get_song_obj(arg_list)

def get_mb_inf(idx):
    mb_inf = musicbrainzngs.get_recording_by_id(idx, includes=['releases', 'artists'])
    return(mb_inf)


# song = network.get_track_by_mbid(i)
# song_tags = song.get_top_tags()


def tptg_prcs(song, mbid):
    """filters tags and weights out of song object"""
    song_tags = song.get_top_tags()
    tag_list = []
    for k in song_tags:
        tag_list.append([mbid, k.item.name, int(k.weight)])
    return(tag_list)
# could partition tags table with weight, that will be a primary thing to sort on 


def save_tags(tags):
    """Saves abbrv-tag-weight list"""
    with open(TAG_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(tags)

        
def insert_failed(mbid):
    "Inserts failed mbids into failed file"
    
    with open(FAILED_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])


def get_todos(chunk_nbr, xisting):
    "Get mbids for which to get tags"
    with open(chunk_dir + chunk_nbr) as fi:
        rdr = csv.reader(fi)
        tags_todo = [i[0] for i in rdr]

    tags_todo2= list(set(tags_todo) - set(xisting))
    
    return(tags_todo2)


def get_xisting(chunk_nbr):
    "Gets mbids that have been checked alread (either done or failed)"
    with open(chunk_dir + chunk_nbr + '_addgs.csv', 'r') as fi:
        rdr = csv.reader(fi)
        xisting = [i[0] for i in rdr]
    return(xisting)    
        

def get_db_songs():
    songs_mbid_abbrev = client.execute('select mbid, abbrv from song_info')

    mbid_abbrv_dict={}

    for i in songs_mbid_abbrev:
        # abbrv_mbid_dict[i[1]]=i[0]
        mbid_abbrv_dict[i[0]]=i[1]

    return(mbid_abbrv_dict)



def save_addgs(mbid, mb_inf, song_orgn, cplt):

    if cplt == 1:
        rls_info = date_prcsr(mb_inf['recording']['release-list'])
        rls_info[0] = rls_info[0].strftime('%Y-%m-%d')

    else:
        rls_info = mb_inf
    
    prntrow = [mbid] + rls_info + [song_orgn]

    with open(ADDGS, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(prntrow)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    # client = Client(host='localhost', password='anudora', database='frrl')
    # mbid_abbrv_dict=get_db_songs()    

    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/'
    FAILED_FILE = chunk_dir + chunk_nbr + '_failed.csv'
    TAG_FILE=chunk_dir + chunk_nbr + '_tags.csv'
    ADDGS = chunk_dir + chunk_nbr + '_addgs.csv'

    API_KEY = "6ff51b99224a1726d47f686d7fcc8083"
    API_SECRET="1ba59bdc2b860b8c9f52ac650e3cb6ab"
    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)
    
    open(FAILED_FILE, 'a')
    open(TAG_FILE, 'a')
    open(ADDGS, 'a')

    xisting = get_xisting(chunk_nbr)
    todos = get_todos(chunk_nbr, xisting)


    for i in todos:
    # for i in failed:
        # song= network.get_track_by_mbid(i)
        try:
            mb_inf = get_mb_inf(i)
        except:
            save_addgs(i, ['mb retrieval failed'], None, 0)
            insert_failed(i)
            continue

        idx = mb_inf['recording']['id']
        # need to handle when mb doesn't find it but lfm does (probably doesn't happen but might)
        # not happening LEL: no mbid -> continue
        # at least for now
        
        a = mb_inf['recording']['artist-credit'][0]['artist']['name']
        t = mb_inf['recording']['title']
        arg_list = [[i], [idx], [a, t]]

        song_output = get_song_obj(arg_list)

        if song_output == 'song not found nowhere':
            print('do stuff here that processes nonfound songs')
            save_addgs(i, ['lfm retrieval failed'], None, 0)
            insert_failed(i)

            # may also have to add condition for no tags, but didn't occur yet

        else:
            song = song_output[1]
            song_orgn = song_output[0]
            
            song_tags = tptg_prcs(song, mbid_abbrv_dict[i])
            save_tags(song_tags)

            save_addgs(i, mb_inf, song_orgn, 1)
            
        if todos.index(i) % 10 ==0:
            print(todos.index(i), len(todos), todos.index(i)/len(todos))


# still need way to process all the other song information
# think easier to save separately as well, and merge later (than to insert into song_info CH directly)
# yup: also allows to run elsewhere much easier

# need to adapt get_dones to work with abbrevs -> needs reverse dict
# nope, just use addgs: addgs should get a line for each song processed
# can use the mlhd id there -> no need for reverse processing 
# might make "failed" superfluous, but think nice to keep at least for testing


# too much shit is missing!
# use musicbrainz to get artist and title 
# might be that many of my current song info's refer to same song
# probably have to clean up logs..
# can make dict with original as key and mb mbid as value
# dicts are quite cheap so might not be an issue
# but means I need to really check all files to be sure there are no duplicates
# need to get extent



###############
# basic setup #
###############

## clickhouseify

# client = Client(host='localhost', password='anudora', database='frrl')
# client.execute('show tables')


# client.execute('create table songs_tags (mbid String, tag String, weight_abs Int32, weight_pct Float32) engine=MergeTree() partition by weight_abs order by tuple()')

# client.execute('create table tags_failed (mbid String, rndm Int32) engine=MergeTree() partition by rndm order by tuple()')

# maybe good idea to split up songs?
# yup: can then be easily distributed

# 250k chunks?

# mbids_ordrd = client.execute('select mbid from song_info3 order by cnt desc')

# cntr = 0
# chunk_cntr = 1
# chunk= []
# for i in mbids_ordrd:
#     chunk.append(i)

#     if cntr == 200000:
#         with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/' + str(chunk_cntr), 'w') as fo:
#             wr = csv.writer(fo)
#             wr.writerows(chunk)

#             chunk = []
#             chunk_cntr +=1
#             cntr = 0
#     cntr+=1


# chunk_cntr +=1
# with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/' + str(chunk_cntr), 'w') as fo:
#             wr = csv.writer(fo)
#             wr.writerows(chunk)





# -> no need to check mbid that lfm produces back in, those always point to themselves


###################################
# some stuff from date processing #
###################################
# ttl_mins = []

# if len(rlss_long) > 0:
#     min_rlss_long = min(rlss_long)
#     ttl_mins.append(min_rlss_long)
# else:
#     ttl_mins.append(datetime.datetime(3000,1,1))

# if len(rlss_medm) > 0:
#     min_rlss_medm = min(rlss_medm)
#     ttl_mins.append(min_rlss_medm)
# else:
#     ttl_mins.append(datetime.datetime(3000,1,1))

# if len(rlss_shrt) > 0:
#     min_rlss_shrt = min(rlss_shrt)
#     ttl_mins.append(min_rlss_shrt)
# else:
#     ttl_mins.append(datetime.datetime(3000,1,1))

# ttl_min = min(ttl_mins)


# NOTES FROM DATE PROCESSING
# cntr =0
# ttl_min_indxs = []
# for i in ttl_mins:
#     if i == ttl_min:
#         ttl_min_indxs.append(cntr)
#     cntr +=1

# min_ttl = min(min(rlss_long), min(rlss_shrt))




# pointless, can't think of it anymore,
# needs to be something more elegant to test whether time issue is present

# tadaa
# level of min > highest level -> problem!
# need to make it clear which level date comes from

# check all buckets in which ttl_min is in
# 


#################
# OLD FUNCTIONS #
#################

# def get_tags(mbid):
    
#     str1="http://ws.audioscrobbler.com/2.0/?method="
#     str2='track' + ".getInfo&mbid="    
#     str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    
    
#     qry = str1 + str2 + mbid + str4

#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()

#     if check_resp(resp_dict, mbid) ==1:
#         mbid_name = resp_dict['track']['name']
        
#         artst_name = resp_dict['track']['artist']['name']
        
#         str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name
#         qry = str1 + str22 + str4

#         resp_raw = requests.get(qry)
#         resp_dict = resp_raw.json()

#         rtrn = resp_dict 
#     else:
#         rtrn = 'wrong'

#     return(rtrn)




# def proc_tags(resp_dict, mbid):
#     """Processes resp dict"""
#     # mbid_name = resp_dict['toptags']['@attr'][mbid_type]
#     # name = resp_dict['toptags']['@attr']['artist']

#     tags = []

#     for i in resp_dict['toptags']['tag']:
#         tags.append([mbid, i['name'], i['count']])
            
#             # do sorting later, better save everything here
#             # all those useless tags might matter too

#     with open(TAG_FILE, 'a') as fo:
#         wr = csv.writer(fo)
#         wr.writerows(tags)

#         # artist name in quotation marks as otherwise commas would break csv
#         # tags can't have commas afaik


# def check_resp(resp_dict, mbid):
#     """Checks a song resp dict from lfm API for all kinds of errors:
#     - song not found
#     - no tags
#     - & in artist or song name 
#     PROBABLY NOT NEEDED ANYMORE
#     """
    
#     if list(resp_dict.keys())[0] == 'error':
#         print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'], mbid)

#         # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
#         insert_failed(mbid)
#         good=0
#         # conn.commit()

#     elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
#         print('oopsie woopsie! no tags!', mbid)
#         insert_failed(mbid)
#         # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
#         good=0

#     elif re.search('&', resp_dict['track']['artist']['name']) is not None:
#         good = 0

#     elif re.search('&', resp_dict['track']['name']) is not None:
#         good = 0

#     else:
#         good=1
#     return(good)




###############
# old testing #
###############


# for i in dones[:50]:    
#     # print(i)
#     # i=dones[47]
#     mb_inf = musicbrainzngs.get_recording_by_id(i, includes=['releases', 'artists'])
#     rls_dt = date_prcsr(mb_inf['recording']['release-list'])
#     print(rls_dt)

#     mb_mbid = mb_inf['recording']['id']

#     mb_title = mb_inf['recording']['title']
#     mb_artst_name = mb_inf['recording']['artist-credit'][0]['artist']['name']

#     lfm_qry_mlhd_id = str1 + str2 + i + str4

#     resp_raw = requests.get(lfm_qry_mlhd_id)
#     resp_dict = resp_raw.json()

#     iss.append(i)

#     try: 
#         lfm_artst_name = resp_dict['track']['artist']['name']
#         lfm_title = resp_dict['track']['name']
#         lfm_mbid = resp_dict['track']['mbid']

#         lfm_mbids.append(lfm_mbid)

#         lfm_qry = str1 + str2 + i + str4
#         resp_raw = requests.get(lfm_qry)
#         resp_dict = resp_raw.json()

#         lfm_id2 = resp_dict['track']['mbid']
#         lfm_mbids2.append(lfm_id2)

        
#     except:
#         lfm_artst_name = 'mlhd id no work'
#         lfm_title = 'mlhd id no work'
#         lfm_mbid = 'mlhd id no work'
#         lfm_mbids.append('mlhd id no work')

#     lfm_qry_mb_mbid = str1 + str2 + mb_mbid + str4
    
#     resp_raw = requests.get(lfm_qry_mb_mbid)
#     resp_dict = resp_raw.json()
    
#     try:
#         mb_lfm_artst_name = resp_dict['track']['artist']['name']
#         mb_lfm_title = resp_dict['track']['name']
#         mb_lfm_mbid = resp_dict['track']['mbid']

#     except:
#         mb_lfm_artst_name = 'mb id no work for lfm'
#         mb_lfm_title = 'mb id no work for lfm'
#         mb_lfm_mbid = 'mb id no work for lfm'
        

#     ################# printing #################
#     print(i)
#     print(mb_mbid)
#     print(lfm_mbid)
#     print(mb_lfm_mbid)
#     print('------------')
#     print(rls_dt)
#     print('------------')
#     print(mb_title)
#     print(lfm_title)
#     print(mb_lfm_title)
#     print('------------')
#     print(mb_artst_name)
#     print(lfm_artst_name)
#     print(mb_lfm_artst_name)
#     print('===============')



########################
# no idea what this is #
########################

# iss = []
# lfm_mbids = []
# lfm_mbids2 = []


# for i in zip(iss, lfm_mbids, lfm_mbids2):
#     print(i[0])
#     print(i[1])
#     print(i[2])    
#     print('--------')

# [print(i[0]==i[1]) for i in zip(lfm_mbids, lfm_mbids2)]



#############
# old tests #
#############



# also two steps, but less danger of interfereence?
# need to tests with failed
# -> can't find failed ones either
# is there a difference between using MB mbid and mb name/title?





# mb_title = mb_inf['recording']['title']
# mb_artst = mb_inf['recording']['artist-credit'][0]['artist']['name']
# song_mlhdid = network.get_track_by_mbid(i)
# song_mbid= network.get_track_by_mbid(idx)

# song2 = network.get_track(mb_artst, mb_title)
# print(i, i==idx, song==song2)

# should i just focus on getting corrected mbid
# or also consider MB name/title? 

# print(song.titlen)


# see if mlhd id and mb id lead to same snog even if not identical
# so far looks like it

# if so: can go with MB by default?
# idk: what if lfm has it but MB not
# also the way still needs two requests for tags
# -> use lfm when possible, MB if lfm breaks, "no song" else

# WRONG: always have to call MB to get release date
# lfm only useful if MB no work
# then i have incomplete information (no release date)
# gonna have incomplete information sometimes anyways (no release dates)

# first MB: has to return MBID and date 
# if fails: date = 'no date'
# try lfm:


# first MB for date
# then mlhdid
# if mlhdid fails, mbid


# with which mbid should i save the tags where the lfm id no works?
# if i use the original:
# easy to get from song log to tags
# if i use mb mbid i first have to make an intermediate step


# are there differences between i and lfm_mbid if i works?
# yup
    

# print(len(mb_inf['recording']['artist-credit']))

# now there can be multiple artists as well WTFFFF
# seems to be mostly one tho

# issue: sometimes the mb_mbid can't be used, have to then use the original mlhd_mbid (i)
# -> need to run both then
# not too difficult tho

# differences in mbids mostly between (1,2) and (3,4)

# exception for dones
# 36e59810-b023-4cd7-ac63-ec4efda7db0c
# 656bc0a1-4506-4a4a-bc1b-924311bd8e1e
# 07457bc3-c8fe-442e-98e5-3e6cae9b7f5b
# 07457bc3-c8fe-442e-98e5-3e6cae9b7f5b


# need to improve for failed
# can happen that neither mlhd nor mb works
# crash when mb mbid not even found
# -> need to be flexible

# need to see in how many cases mb mbid works but mlhd does not

# print('----------------------------------------------')
    
# musicbrainzngs.get_recording_by_id(mbid2)['recording']['id']

# musicbrainzngs.get_recording_by_id('8fed0a52-988a-4701-8219-ca395400caae')


# mbid='27608c0c-a89b-402d-a809-5748e72a5e82'
# mbid='82a07882-13cf-4ea3-9d6c-6dd66e87fe63'
# ----------------------------------------------
# mbid='36e59810-b023-4cd7-ac63-ec4efda7db0c'
# mbid='656bc0a1-4506-4a4a-bc1b-924311bd8e1e'
# mbid='07457bc3-c8fe-442e-98e5-3e6cae9b7f5b'
# ----------------------------------------------
# 457d0b8d-7970-4783-a480-6fea64596502
# ec9ad7d2-a12d-4e33-9404-55eae3e9b619

# i think i need more song information
# mbid mlhd
# mbid mb
# mbid lfm

# name mb
# name lfm

# artist name mb
# artist name lfm

# album
# - name mb
# - 
# album is 

# release data would be nice
# but there are 20 of those
# want to get first
# but some just have year
# if youngest is just a year and i set that to year-12-31
# i can get spike at new years
# add resolution of date, allows to filter more 

# need to check for how many that's the case

# rls_lst = x['recording']['release-list']

# time difference between min_ttl and min of highest res? 



#####################################
# less old tests but still unneeded #
#####################################


# SETUP on runtime
# chunk_nbr = '1'

# i = dones[69]
# i = 'ca58a9fb-66c2-4589-bca9-4f9494905018'

# for i in dones[240:250]:
# # for i in failed[0:20]:
#     try:
        
#         # song= network.get_track_by_mbid(i)
#         mb_inf = get_mb_inf(i)
#         idx = mb_inf['recording']['id']

#         song_output = first([i, idx])
        
#         if song_output == 'song not found nowhere':
#             print('do stuff here that processes nonfound songs')
#             # may also have to add condition for no tags

#         else:
#             song = song_output[1]
#             song_orgn = song_output[0]
            
#             song_tags = tptg_prcs(song, i)

#     except:
#         print('song not found')
#         probs.append((i, idx))
#         pass




# for i in failed:
#     try:
#         x = get_mb_inf(i)
#     except:
#         print('no mibd')

#     try:
#         song=network.get_track(x['recording']['artist-credit'][0]['artist']['name'], x['recording']['title'])
# # network.get_track_by_mbid(i)
        
#     except:
#         print('no lfm id')

#     print('----------')



###################################
# more recent but still old funcs #
###################################


# def first(flist, default=None):
#     org_dict = {1:'lfm', 2:'mb'}

#     cntr = 1
#     for f in flist:
#         try:
#             sx = network.get_track_by_mbid(f)
#             return([org_dict[cntr], sx])
#         except:
#             cntr +=1
#             continue
        
#     else:
#         # maybe i should also return a list here
#         return 'song not found nowhere'


# def get_xisting(chunk_nbr):
#     "Gets mbids that have been checked alread (either done or failed)"
#     with open(chunk_dir + chunk_nbr + '_failed.csv', 'r') as fi:
#         rdr = csv.reader(fi)
#         failed = [i[0] for i in rdr]

#     with open(chunk_dir + chunk_nbr + '_tags.csv', 'r') as fi:
#         rdr = csv.reader(fi)
#         dones_raw = [i[0] for i in rdr]
#         dones = list(set(dones_raw))

#         xisting = dones + failed
#         return(xisting)
