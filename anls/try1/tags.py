import csv
import requests

def check_existing():
    with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tags.csv', 'r') as fi:
        rdr = csv.reader(fi)
        exstng_artsts = [row[0] for row in rdr]
    return(exstng_artsts)



def get_tags(artst):
    
    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2="artist.getTopTags&mbid="
    str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

    # artst = '936db8ea-2d4b-4d2b-9665-ad8fdecac835'
    qry = str1 + str2 + artst + str4
    
    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    if list(resp_dict.keys())[0] == 'error':
        print('oopsie woopsie!')

    else: 
        name = resp_dict['toptags']['@attr']['artist']

        tags = [artst,name]
        for i in resp_dict['toptags']['tag']:
            # tag_weight_pr = (i['name'],i['count'])
            tags.append(i['name'])
            tags.append(i['count'])

            # artist name in quotation marks as otherwise commas would break csv
            # tags can't have commas afaik

        with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tags.csv', 'a') as fo:
            wr = csv.writer(fo)
            wr.writerow(tags)


with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/mbid_list.csv', 'r') as fi:
    rdr = csv.reader(fi)
    mbids_to_do = [row[0] for row in rdr]

mbids_done = check_existing()
mbids_to_do2 = list(set(mbids_to_do) - set(mbids_done))


cntr = 0
for i in mbids_to_do2:

    get_tags(i)

    if cntr ==25:
        print(mbids_to_do2.index(i))
        cntr = 0
    cntr+=1



