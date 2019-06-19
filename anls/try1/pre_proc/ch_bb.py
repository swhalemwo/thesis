# clickhouse bulk bulk (bb)

import os

paths= [
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/00/',
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/01/',
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/02/',
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/03/',
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/04/',
    '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/05/',
    '/home/johannes/Downloads/mlhd/06/',
    '/home/johannes/Downloads/mlhd/07/',
    '/home/johannes/Downloads/mlhd/08/',
    '/home/johannes/Downloads/mlhd/09/']

for p in paths:
    print(p)
    os.system('python3.6 ~/Dropbox/gsss/thesis/anls/try1/pre_proc/read_in_ch_blk.py ' + p)
