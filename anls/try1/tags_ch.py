import csv
import requests
import argparse
# from clickhouse_driver import Client
import os
import re
from calendar import monthrange

# mbid = '94a2a5ba-390d-4dd9-9f69-3e314f8b9d98'
# mbid= '0e37764b-0726-4a15-8c16-abc07c4ea033'

# mbid = i
mbid = failed[300]

mbid2 = musicbrainzngs.get_recording_by_id(failed[300])['recording']['id']
mbid = mbid2

# for i in dones[:50]:
for i in failed[:50]:    
    # print(i)
    mb_inf = musicbrainzngs.get_recording_by_id(i, includes=['releases', 'artists'])
    rls_dt = date_prcsr(mb_inf['recording']['release-list'])
    mb_mbid = mb_inf['recording']['id']

    mb_title = mb_inf['recording']['title']
    mb_artst_name = mb_inf['recording']['artist-credit'][0]['artist']['name']

    lfm_qry_mlhd_id = str1 + str2 + i + str4

    resp_raw = requests.get(lfm_qry_mlhd_id)
    resp_dict = resp_raw.json()

    try: 
        lfm_artst_name = resp_dict['track']['artist']['name']
        lfm_title = resp_dict['track']['name']
        lfm_mbid = resp_dict['track']['mbid']
    except:
        lfm_artst_name = 'mlhd id no work'
        lfm_title = 'mlhd id no work'
        lfm_mbid = 'mlhd id no work'

    lfm_qry_mb_mbid = str1 + str2 + mb_mbid + str4
    
    resp_raw = requests.get(lfm_qry_mb_mbid)
    resp_dict = resp_raw.json()
    
    try:
        mb_lfm_artst_name = resp_dict['track']['artist']['name']
        mb_lfm_title = resp_dict['track']['name']
        mb_lfm_mbid = resp_dict['track']['mbid']

    except:
        mb_lfm_artst_name = 'mb id no work for lfm'
        mb_lfm_title = 'mb id no work for lfm'
        mb_lfm_mbid = 'mb id no work for lfm'
        

    ################# printing #################
    print(i)
    print(mb_mbid)
    print(lfm_mbid)
    print(mb_lfm_mbid)
    print('------------')
    print(rls_dt)
    print('------------')
    print(mb_title)
    print(lfm_title)
    print(mb_lfm_title)
    print('------------')
    print(mb_artst_name)
    print(lfm_artst_name)
    print(mb_lfm_artst_name)
    print('===============')
    
    

# print(len(mb_inf['recording']['artist-credit']))

# now there can be multiple artists as well WTFFFF
# seems to be mostly one tho

# issue: sometimes the mb_mbid can't be used, have to then use the original mlhd_mbid (i)
# -> need to run both then
# not too difficult tho

differences in mbids mostly between (1,2) and (3,4)

exception for dones
36e59810-b023-4cd7-ac63-ec4efda7db0c
656bc0a1-4506-4a4a-bc1b-924311bd8e1e
07457bc3-c8fe-442e-98e5-3e6cae9b7f5b
07457bc3-c8fe-442e-98e5-3e6cae9b7f5b


need to improve for failed
can happen that neither mlhd nor mb works
crash when mb mbid not even found
-> need to be flexible

need to see in how many cases mb mbid works but mlhd does not



ll1=["07b81a14-106d-4dbe-b27d-2545a0d17268","07b2a832-05f3-45b9-a497-55bcdbd1b4e8","466e5582-6f79-4acd-91f0-c39f0bd46bf6","876058ae-8354-4492-a237-9ef9f0b3e6b7","0a392f1f-61b7-43bd-81f1-de7789d1d26c","2b783759-4cf1-462c-8810-312e04276533","73be27de-1be1-4f16-850e-a8ec35d9a1e1","09b40797-8c14-4867-a59d-1495655685ab","036cb655-c6d6-4dc0-8b98-1cd45cd9cee2","d57d9536-a186-413d-9df4-6723ee5577a6","c2bdabae-8f35-456b-bf9b-68e861da7988"]

ll2=["s141","s145","s146","s181","s194","s219","s232","s244","s245","s259","s266"]


ll1_long = []
ll2_long = []

for i in range(1000):
    nbr = random.sample(range(11),1)[0]

    ll1_long.append(ll1[nbr])
    ll2_long.append(ll2[nbr])


str1="http://ws.audioscrobbler.com/2.0/?method="
str2='track' + ".getInfo&mbid="    
str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"




print('----------------------------------------------')
    
musicbrainzngs.get_recording_by_id(mbid2)['recording']['id']

musicbrainzngs.get_recording_by_id('8fed0a52-988a-4701-8219-ca395400caae')


mbid='27608c0c-a89b-402d-a809-5748e72a5e82'
mbid='82a07882-13cf-4ea3-9d6c-6dd66e87fe63'
----------------------------------------------
mbid='36e59810-b023-4cd7-ac63-ec4efda7db0c'
mbid='656bc0a1-4506-4a4a-bc1b-924311bd8e1e'
mbid='07457bc3-c8fe-442e-98e5-3e6cae9b7f5b'
----------------------------------------------
457d0b8d-7970-4783-a480-6fea64596502
ec9ad7d2-a12d-4e33-9404-55eae3e9b619

i think i need more song information
mbid mlhd
mbid mb
mbid lfm

# name mb
# name lfm

# artist name mb
# artist name lfm

album
- name mb
- 
album is 

release data would be nice
but there are 20 of those
want to get first
but some just have year
if youngest is just a year and i set that to year-12-31
i can get spike at new years

# need to check for how many that's the case


rls_lst = x['recording']['release-list']
rls_lst = mb_inf['recording']['release-list']


date_prcsr(rls_lst)

def date_prcsr(rls_lst):

    rlss_long=[]
    rlss_medm=[]
    rlss_shrt=[]

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
        min_ttl = 'no date'
    else:
        all_dts = rlss_long + rlss_medm + rlss_shrt

        min_ttl = min(all_dts)

        ttl_mins = []

        if len(rlss_long) > 0:
            min_rlss_long = min(rlss_long)
            ttl_mins.append(min_rlss_long)
        else:
            ttl_mins.append(datetime.datetime(3000,1,1))

        if len(rlss_medm) > 0:
            min_rlss_medm = min(rlss_medm)
            ttl_mins.append(min_rlss_medm)
        else:
            ttl_mins.append(datetime.datetime(3000,1,1))

        if len(rlss_shrt) > 0:
            min_rlss_shrt = min(rlss_shrt)
            ttl_mins.append(min_rlss_shrt)
        else:
            ttl_mins.append(datetime.datetime(3000,1,1))

        ttl_min = min(ttl_mins)

        buckets = [rlss_shrt, rlss_medm, rlss_long]

        ttl_min_mbrshps = []
        cntr = 1
        for i in buckets:
            if ttl_min in i:
                ttl_min_mbrshps.append(cntr)

            cntr +=1

        if max(ttl_min_mbrshps) > lowest_lvl:
            print("DATE ISSUE")
            # maybe add some return thing

    return(min_ttl)

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



def get_tags(mbid):
    
    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2='track' + ".getInfo&mbid="    
    str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    
    
    qry = str1 + str2 + mbid + str4

    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    if check_resp(resp_dict, mbid) ==1:
        mbid_name = resp_dict['track']['name']
        
        artst_name = resp_dict['track']['artist']['name']
        
        str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name
        qry = str1 + str22 + str4

        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()

        rtrn = resp_dict 
    else:
        rtrn = 'wrong'

    return(rtrn)

def proc_tags(resp_dict, mbid):
    # mbid_name = resp_dict['toptags']['@attr'][mbid_type]
    # name = resp_dict['toptags']['@attr']['artist']

    tags = []

    for i in resp_dict['toptags']['tag']:
        tags.append([mbid, i['name'], i['count']])
            
            # do sorting later, better save everything here
            # all those useless tags might matter too

    with open(TAG_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(tags)

        # artist name in quotation marks as otherwise commas would break csv
        # tags can't have commas afaik


def check_resp(resp_dict, mbid):
        if list(resp_dict.keys())[0] == 'error':
            print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'], mbid)

            # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            insert_failed(mbid)
            good=0
            # conn.commit()

        elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
            print('oopsie woopsie! no tags!', mbid)
            insert_failed(mbid)
            # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            good=0

        elif re.search('&', resp_dict['track']['artist']['name']) is not None:
            good = 0

        elif re.search('&', resp_dict['track']['name']) is not None:
            good = 0
            
        else:
            good=1
        return(good)

def insert_failed(mbid):
    with open(FAILED_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])


def get_todos(chunk_nbr, xisting):

    with open(chunk_dir + chunk_nbr) as fi:
        rdr = csv.reader(fi)
        tags_todo = [i[0] for i in rdr]

    tags_todo2= list(set(tags_todo) - set(xisting))
    
    return(tags_todo2)

def get_xisting(chunk_nbr):
    with open(chunk_dir + chunk_nbr + '_failed.csv', 'r') as fi:
        rdr = csv.reader(fi)
        failed = [i[0] for i in rdr]

    with open(chunk_dir + chunk_nbr + '_tags.csv', 'r') as fi:
        rdr = csv.reader(fi)
        dones_raw = [i[0] for i in rdr]
        dones = list(set(dones_raw))

        xisting = dones + failed
        return(xisting)

# SETUP on runtime


# chunk_nbr = '1'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    
    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/'
    FAILED_FILE = chunk_dir + chunk_nbr + '_failed.csv'
    TAG_FILE=chunk_dir + chunk_nbr + '_tags.csv'
    
    open(FAILED_FILE, 'a')
    open(TAG_FILE, 'a')

    xisting = get_xisting(chunk_nbr)

    todos = get_todos(chunk_nbr, xisting)

    for i in todos:
        resp_dict=get_tags(i)
        if type(resp_dict) == type({}):
            proc_tags(resp_dict, i)

        if todos.index(i) % 10 ==0:
            print(todos.index(i), len(todos), todos.index(i)/len(todos))

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













from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


##### Example call #####

# if __name__ == '__main__':
#     d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
#     print(total_size(d, verbose=True))


print(total_size(ll1_long))
print(total_size(ll2_long))

    
