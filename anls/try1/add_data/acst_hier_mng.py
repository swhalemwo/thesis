import csv
import itertools
import os
from time import sleep

coefs = [0.75, 1, 1.25]

combs = list(itertools.product(coefs, repeat=2))

basedir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust/'

for i in combs:
    print(i, combs.index(i))
    try:

        fldr = 'harsh_' +str(float(i[0])) + "_tp_" + str(float(i[1]))
        base_folders = os.listdir(basedir)

        if fldr in base_folders:
            # check if folder already exists
            fld_files = os.listdir(basedir + fldr)
            if "debug.csv" in fld_files:
                # and if any runs have been made
                with open(basedir + fldr + '/debug.csv', 'r') as fi:
                    rdr = csv.reader(fi)
                    tps = [int(r[1]) for r in rdr]

                max_tp = int(max(tps))
                tp_start = max_tp +1
            else:
                # debug.csv not there: broken on first run, just start again
                tp_start = 0

        else:
            # create folder
            fldr_crt = 'mkdir ' + basedir + fldr
            os.system(fldr_crt)
            tp_start = 0

        print(tp_start)
        
        ex_str = "python3.6 acst_hier.py " + str(i[0]) + " " + str(i[1])+ " " + str(tp_start)
        print(ex_str)
        os.system(ex_str)

    except:
        
        pass

