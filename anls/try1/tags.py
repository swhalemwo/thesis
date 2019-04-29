import csv
import requests
import argparse

class tag_dldr():
    def check_existing():
        with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tags.csv', 'r') as fi:
            rdr = csv.reader(fi)
            exstng_artsts = [row[0] for row in rdr]
        return(exstng_artsts)


    def load_todo(infile)
        with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/mbid_list.csv', 'r') as fi:
            rdr = csv.reader(fi)
            mbids_to_do = [row[0] for row in rdr]



    def get_tags(mbid, mbid_type):

	# mbid='2d1d13bc-a6be-4d50-a67a-ec89bb8ddd46'
        # mbid='3cd374d4-1ead-4a93-9e55-330bc896eb64'
        # mbid='3f173d9c-bcb2-440d-9ccb-92288546c0aa'
        # mbid = '08a66a67-63cb-4598-8bdc-28dcaae34350'


        # rage against the machine
        # mbid = '3798b104-01cb-484c-a3b0-56adc6399b80'
        mbid = '10cf6ada-ef3f-3cd3-9a7b-bdc15db2dddf'

        # 1197129695	3798b104-01cb-484c-a3b0-56adc6399b80	10cf6ada-ef3f-3cd3-9a7b-bdc15db2dddf	fa2e0148-835d-473a-babf-c2a3188879d2
        
        # https://musicbrainz.org/ws/2/url/ws/2/release?label=10cf6ada-ef3f-3cd3-9a7b-bdc15db2dddf
        # https://musicbrainz.org/ws/2/url/ws/2/artist?label=3798b104-01cb-484c-a3b0-56adc6399b80

        # https://musicbrainz.org/ws/2/url/ws/2/recording?label=fa2e0148-835d-473a-babf-c2a3188879d2

        # https://musicbrainz.org/ws/2/url/fa2e0148-835d-473a-babf-c2a3188879d2

        # https://musicbrainz.org/ws/2/url/ws/2/artist/3798b104-01cb-484c-a3b0-56adc6399b80

        mbid_type = 'artist'
        mbid_type = 'album'
        mbid_type = 'track'


        str1="http://ws.audioscrobbler.com/2.0/?method="
        str2=mbid_type +".getTopTags&mbid="
        str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

        # artst = '936db8ea-2d4b-4d2b-9665-ad8fdecac835'
        qry = str1 + str2 + mbid + str4

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


mbids_done = check_existing()
mbids_to_do2 = list(set(mbids_to_do) - set(mbids_done))


cntr = 0
for i in mbids_to_do2:

    get_tags(i)

    if cntr ==25:
        print(mbids_to_do2.index(i))
        cntr = 0
    cntr+=1



# parameters to add:
- type: artist, album, song
- infile: list of mbid
- outfile: result file
