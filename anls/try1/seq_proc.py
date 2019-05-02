import csv
import time

from os import listdir
import graph_tool as gt
import subprocess
import os, sys
from datetime import datetime


from graph_tool.all import *
from graph_tool import *  



stage_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/stage/"
stage_files = listdir(stage_dir)


i = stage_files[0]

gx = gt.load_graph_from_csv(stage_dir+i, directed=True, string_vals=True, csv_options={'delimiter':'\t'})
rel_mbids = find_vertex_range(gx, 'in', (800, 100000000))
# need to set limits

# niche: contribution should be based on in degree

tasks:
- album mbid lists
- mds
- niche size
  need to specify tags
  -> need to know uses of tags per year


