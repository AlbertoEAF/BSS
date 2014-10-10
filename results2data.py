#!/usr/bin/env python3

import sys
import os
import subprocess as sub
#import itertools
import re

from ConfigParser import *
from color_codes import *

def dirs(folder):
    return [ d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d)) ]

def listdir_endfilter(folder, filtr): 
    """ use to search for extensions for e.g. """
#    for f in os.listdir(folder):
#        print(f)
#        print(f[-len(filtr):])
    return [ f for f in os.listdir(folder) if f[-len(filtr):] == filtr ]
    

def subdirs_file_endfilter(folder, filtr):
    """ Takes a folder and looks for the file inside each it's 1-level-down subdirectories that matches the end filter filtr. """
    results_dict = {}
    for d in dirs(folder):
        subd = os.path.join(folder,d)
        files =  listdir_endfilter(subd, filtr) 
        if len(files) > 1:
            error("Multiple files in one directory when searching: <{}>".format(filtr))
        results_dict[d] = files[0]
    return results_dict

def results2data(folder, output_path, extension):
    """ Merges all .iresults in subfolders into a single file for gnuplot.  Adds 2 columns. The first is the name of the subfolder of the test and the second the id of the test (first numeric substring found in that name).  """
    results = subdirs_file_endfilter(folder, extension) 
   
    header = None
    rows = []
    for d in results:
        lines = []
        with open( os.path.join(folder,d,results[d]) ) as f:
            lines = f.readlines()

        if not header:
            header = ['name','id'] + lines[0].split()

        id = re.findall(r'-?\d+', d)[0]
        rows.append([d,id]+lines[1].split())

    rows.sort(key=lambda x: int(x[1]) ) # sort by id
    
    with open(output_path,'w') as data:
        data.write("\t".join(header)+"\n")
        for r in rows:
            data.write("\t".join(r)+"\n")

def degeneracies2data(folder, output_path):
    """ Merges all .iresults in subfolders into a single file for gnuplot. """
    results = subdirs_file_endfilter(folder, ".degeneracies") 
   
    data = open(output_path,'w')

    columns = 0
    header_printed = False
    for d in results:
        lines = []
        with open( os.path.join(folder,d,results[d]) ) as f:
            lines = f.readlines()

        if not header_printed:
            header_printed = True
            columns = len(lines[0].split())
            data.write("#name\tid\t")       # Name and id columns.
            for i in range(columns):
                data.write(str(i)+" ")     # Number of degeneracies column headers.
            data.write("\n")

        id = re.findall(r'-?\d+', d)[0]
        
        data.write(d+"\t"+id+"\t") # Name column.
        data.write(lines[0])               # Positive degeneracy values,
        data.write(d+"\t"+id+"\t") # Name column (same index, negative values).
        data.write(lines[1])               # Negative degeneracy values.


    data.close()
    

if __name__ == "__main__": # Usage: prgm <folder>
    results2data     ( sys.argv[1], os.path.join(sys.argv[1],"Results"     ), ".results" ) 
    results2data     ( sys.argv[1], os.path.join(sys.argv[1],"Iresults"    ), ".iresults") 
    degeneracies2data( sys.argv[1], os.path.join(sys.argv[1],"Degeneracies")             )
