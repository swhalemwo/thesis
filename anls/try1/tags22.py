
import pylast
API_KEY = "6ff51b99224a1726d47f686d7fcc8083"
API_SECRET="1ba59bdc2b860b8c9f52ac650e3cb6ab"
# network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

mbid='b6ab0f53-9acf-430b-b3d8-8b79b0a28201'




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


def check_resp(resp_dict, mbid):
    """Checks a song resp dict from lfm API for all kinds of errors:
    - song not found
    - no tags
    - "&" in artist or song name 
    """
    
    if list(resp_dict.keys())[0] == 'error':
        print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'], mbid)

        insert_failed(mbid)
        good=0
        # conn.commit()

    elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
        print('oopsie woopsie! no tags!', mbid)
        insert_failed(mbid)
        good=0

    elif re.search('&', resp_dict['track']['artist']['name']) is not None:
        good = 0

    elif re.search('&', resp_dict['track']['name']) is not None:
        good = 0

    else:
        good=1
    return(good)


def proc_tags(resp_dict, mbid):
    """Processes resp dict"""
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


################################
# pylast vs web API comparison #
################################

# t1=time.time()
# for i in range(20):
#     sx = network.get_track_by_mbid(mbid)
#     tgs = sx.get_top_tags()
#     print(i)
# t2=time.time()


# t1=time.time()
# for i in range(20):
#     str1="http://ws.audioscrobbler.com/2.0/?method="
#     str2='track' + ".getInfo&mbid="    
#     str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

#     qry = str1 + str2 + mbid + str4

#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()

#     mbid_name = resp_dict['track']['name']
#     artst_name = resp_dict['track']['artist']['name']

#     str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name
#     qry = str1 + str22 + str4    
    
#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()
#     print(i)
    
# t2=time.time()

# hm using web api is twice as fast
# question kinda is if it is less accurate
# have to rely on getting top tags with artist name and title
# can't figure out how pylast works

# also playcount and listener count would be nice info to have: sees how well represented my sample is for last fm standards

# aaaaa
