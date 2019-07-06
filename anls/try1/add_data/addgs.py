


data_dir = '/home/johannes/mega/gsss/thesis/remotes'

xshellz=['chunk4/','chunk7/','chunk11/','chunk14/']

# collapse tree? so that duplicate stuff doesn't fuck up stuff
# de-duplicating is not expensive tho, only few seconds for millions of tags
# and only needed once

# ** preps
# *** chunk4
# copying/catting tag files together manually 
addgs_fldr = 'v5'
tags_fl = '/home/johannes/mega/gsss/thesis/remotes/chunk4/readin/4_tags_fnl.csv'

# dfx = pd.read_csv(tags_fl, names = ['lfm_id', 'tag', 'weight'])

# *** chunk7

# figure out what files to use
# tags_fl = ['/home/johannes/mega/gsss/thesis/remotes/chunk7/readin/7_tags_'+str(i) +'.csv' for i in range(1,5)]

# sets7 = []
# for tag_fl in tags_fl:
#     dfx = pd.read_csv(tag_fl, names = ['lfm_id', 'tag', 'weight'])
#     xset = set(dfx['lfm_id'])
#     sets7.append(xset)

# for x in sets7:
#     for y in sets7:
#         print(sets7.index(x), sets7.index(y), len(x & y))

# now just use v2 for addgs, and the tags of 3 and 4 for final (catted together manually)

addgs_fldr = 'v2'
tags_fl = '/home/johannes/mega/gsss/thesis/remotes/chunk7/readin/7_tags_fnl.csv'

# *** chunk11

addgs_fldr = 'v2'
tags_fl = '/home/johannes/mega/gsss/thesis/remotes/chunk11/readin/11_tags_fnl.csv'

# *** chunk14

addgs_fldr = 'v2'
tags_fl = '/home/johannes/mega/gsss/thesis/remotes/chunk14/readin/14_tags_fnl.csv'

## ** overall prep
xshellz={'chunk4/': {'ad_fl':'v5/', 'tag_fl':'/home/johannes/mega/gsss/thesis/remotes/chunk4/readin/4_tags_fnl.csv'},
         'chunk7/': {'ad_fl':'v2/', 'tag_fl':'/home/johannes/mega/gsss/thesis/remotes/chunk7/readin/7_tags_fnl.csv'},
         'chunk11/':{'ad_fl':'v2/', 'tag_fl':'/home/johannes/mega/gsss/thesis/remotes/chunk11/readin/11_tags_fnl.csv'},
         'chunk14/':{'ad_fl':'v2/', 'tag_fl':'/home/johannes/mega/gsss/thesis/remotes/chunk14/readin/14_tags_fnl.csv'}}
    

         ,'chunk7/','chunk11/','chunk14/']

# * preppare basic files

base_path = '/home/johannes/mega/gsss/thesis/remotes/'
chunk_dirs = ['chunk' + str(i) + '/' for i in range(1,16)] + ['chunk17/']

ttl_path = '/home/johannes/mega/gsss/thesis/remotes/ttl/'

for i in chunk_dirs:
    if i in xshellz.keys():
        
        addgs_file = base_path + i + xshellz[i]['ad_fl'] + i + i[5:len(i)-1] + '_addgs.csv'
        tag_file = base_path + i + 'readin/' + i[5:len(i)-1] + '_tags_fnl.csv'
        dones_tags = base_path + i + xshellz[i]['ad_fl'] + i + i[5:len(i)-1] + '_dones_tags.csv'

        pass
    else:
        addgs_file = base_path + i + i[5:len(i)-1] +'_addgs.csv'
        tag_file = base_path + i + i[5:len(i)-1] +'_tags.csv'
        dones_tags = base_path + i + i[5:len(i)-1] +'_dones_tags.csv'

    # print(addgs_file)
    # print(tag_file)
    # print(addgs_file)
    
    addgs_str = 'cat ' + addgs_file + ' >> ' + ttl_path + 'ttl_addgs.csv'
    tags_str = 'cat ' + tag_file + ' >> ' + ttl_path + 'ttl_tags.csv'
    dones_tags_str = 'cat ' + dones_tags + ' >> ' + ttl_path + 'ttl_dones_tags.csv'
    
    # os.system(addgs_str)
    # os.system(tags_str)
    os.system(dones_tags_str)
        

# * read in
# ** tags:
# use ~/Dropbox/gsss/thesis/anls/try1/tag_insert_ch.py
# DONE 
# ** addgs
# use ~/Dropbox/gsss/thesis/anls/try1/add_data/ch_setups.py

# ** dones_tags
