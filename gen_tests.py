#! /usr/bin/python3

### usage: prgm <test_file.test>

from os import listdir
from os.path import isfile, join
import os
import subprocess as sub
import itertools
import re

import math

from ConfigParser import *
from color_codes import *
from parse_run import *

def degrees2rad(degrees): 
    return math.pi/180.0 * degrees

def rad2degrees(rad): 
    return 180.0/math.pi * rad


def rm_extension(path):
    if "." in path:
        return path[:path.rfind(".")]
    else:
        return path

def listdir_waves(dirpath):
    return [ f for f in listdir(dirpath) if f[-4:]==".wav" ]

def get_wav_files(dirpath, pattern):
    """ Returns the relative paths of the .wav files in a dir that match the regexp pattern. """
    
    all_files = listdir_waves(dirpath)

    files = []

    for f in all_files:
        if re.match(pattern, f):
            #files.append(dirpath+'/'+f)
            files.append(f)
            

    return files

def print_wavlist(name, dirpath, full_list, partial_list):
    print(YELLOW,name," @ ", dirpath, ": ",NOCOLOR, sep="", end="")
    for f in full_list:
        if f in partial_list:
            print(CYAN,f,NOCOLOR, sep="", end=" ")
        else:
            print(f, end=" ")
    print(YELLOW,"(",len(partial_list),"/",len(full_list),")\n",NOCOLOR, sep="")

def exclude(f, f_ref, exclusion_rules):
    """ Implemented only for the format GnN.wav where G is the gender and n the number of the speaker of that gender thus the subject is identified with Gn and N is the sentence, where both G and n take one character. """

    excl = False

    if 'same' in exclusion_rules:
        if f == f_ref:
            excl = True
    if 'same_sex' in exclusion_rules:
        if f[0] == f_ref[0]:
            excl = True
    if 'same_subject' in exclusion_rules:
        if f[:2] == f_ref[:2]:
            excl = True
    if 'same_sentence' in exclusion_rules:
        if f[2:] == f_ref[2:]:
            excl = True

    return excl


def gen_combinations(test_file):
    """
    Takes a .test file and generates the necessary .csim files or runs the tests if mix is the command 
    """

    test = ConfigParser(test_file, "parent_localpath")

    rules = test['exclusion_rules'].split()


    sources = [ test['sources_folder']+"/"+f for f in test['sources'].split() ]
    sources = [ s.replace("//","/") for s in sources ]

    N = len(sources)


    print(YELLOW, "\nN =", N, NOCOLOR, "\n")

    dirs  = []
    files = []

    for n in range(N):
        dirpath = sources[n][:sources[n].rfind('/')]
        pattern = sources[n][sources[n].rfind('/')+1:]

        dirs.append( dirpath )
        files.append( get_wav_files(dirpath, pattern) )

        print_wavlist("s"+str(n),dirpath,listdir_waves(dirpath),files[n])

    all_combinations = itertools.product(*files)
    combinations = []
    for combination in all_combinations:
        exclude_flag = False
        for n1 in range(N-1):
            for n2 in range(n1+1,N):
                if exclude(combination[n1], combination[n2], rules):
                    exclude_flag = True

        if not exclude_flag:
            combinations.append(combination)

    return (N,dirs,combinations)


def execute_ecoduet_and_bss_eval_combi(folder,test,c,dirs,N,duetcfg):
    combi_name = "_".join([ c[n][:-4] for n in range(N) ])

    ecolog  = folder+combi_name+".ecolog"
    ecologi = folder+combi_name+".ecologi" # ibm

    bsslog  = folder+combi_name+".bsslog"
    bsslogi = folder+combi_name+".bsslogi" # ibm

    print("Running ecoduet...",end="",flush=True)

    out = None
    duet_flags=""
    try:
        duet_flags=test['duet_flags']
        print("<",duet_flags,">",end="",sep="",flush=True)
        out = sub.check_output(['r','-l', ecolog,'-i', ecologi]+ duet_flags.split() + [duetcfg])
    except KeyError:
        out = sub.check_output(['r','-l', ecolog,'-i', ecologi, duetcfg])
    print("OK")


    if test.i('check_degeneracy'):
        print("Checking degeneracy through ecoduet.log...",end="",flush=True)
        N_,Ne_,_,_,_,_,_ = parse_ecoduet(ecolog)
        if N_ == Ne_:
            print("OK")
        else:
            error("FAIL! N={} != Ne={}".format(N_,Ne_))

    if (test.i('disable_bss_eval')):
        print(RED,"Skipping BSS Eval.",NOCOLOR,sep="")
    else:
        print("BSS Eval async call...", end="",flush=True)
        exec_bss_eval_static_and_ibm(bsslog, bsslogi, test['skip_time'])
#                sub.check_call(["stty","sane"])
        print("OK")





def test(test_file):
    """
    Run all tests and save the logs for later processing.

    ecoduet saves 2 logs: ecoduet.log and ecoduet_ibm.log
    similarly we'll call bss_eval to output: bss_eval.log and bss_eval_ibm.log
    """
    
    folder = test_file[:test_file.rfind("/")+1]

    test = ConfigParser(test_file, "parent_localpath")
    (N,dirs,combinations) = gen_combinations(test_file)

    sub.check_call(["cleantests.sh",folder])

    try:
        duetcfg = test['duet_cfg']
    except KeyError:
        cfgs = [ f for f in os.listdir(folder) if f[-4:]==".cfg" ]
        if len(cfgs) > 1:
            error("More than a .cfg locally available. Please set duet_cfg in the .test file.")
        duetcfg = (folder+"/"+cfgs[0]).replace("//","/")
        print("duet_cfg not set in test. Using local .cfg:", duetcfg)

    sub.check_call(["cleantests.sh",folder])

    if test['mixer'] == 'mix':
        for i_c in range(len(combinations)):
            c = combinations[i_c]
            print( GREEN, "Testing({}/{}): {}".format(i_c+1,len(combinations),c) , NOCOLOR, sep="")

            sub.check_call(['mix']+[ dirs[n]+'/'+c[n] for n in range(N) ])

            execute_ecoduet_and_bss_eval_combi(folder,test,c,dirs)

    elif test['mixer'] == 'csim':
        if combinations: # Otherwise don't erase old results.
            sub.check_call(["rm","-f",folder+"*.csim"])
        for i_c in range(len(combinations)):
            c = combinations[i_c]
            print( GREEN, "Testing({}/{}): {}".format(i_c+1,len(combinations),c) , NOCOLOR, sep="")

            sources_pos = test['sources_pos'].split()
            if "m" in sources_pos[0]: 
                # sources_pos in format : <distance1>m<degrees1>o ...
                if len(sources_pos) != len(c):
                    error("ERROR: Wrong number of sources positions in <distance>m<angle>o format.")
            elif len(sources_pos) != len(c)*3:
                # sources_pos in format : <x1> <y1> <z1> ...
                error("ERROR: Wrong number of sources positions in <x> <y> <z> format.")

            path_csim = folder+"_".join([ rm_extension(c[n]) for n in range(N) ]) + ".csim"
            with open(path_csim,'w') as csim:
                csim.write("sources_folder = "+test['sources_folder']+"\n")
                csim.write("output_folder = sounds\n\n")
                csim.write("Delta = "+test['Delta']+"\n")
                csim.write("c = "+test['c']+"\n\n")
                for n in range(N):
                    dir_n = dirs[n][len(test['sources_folder'])+1:] 
                    if (dir_n):
                        dir_n += '/'

                    # Source position (convert from <d>m<phi>o to x y z if needed).
                    sx = sy = sz = ""
                    if "m" in sources_pos[0]:
                        s = sources_pos[n]
                        d,angle = re.search('(\d+\.?\d*)m(-?\d+\.?\d*)o$', s).groups()
                        d = float(d)
                        angle = float(angle)
                
                        # Coordinate system where the angle refers to the y axis.
                        sx = str( d*math.sin(degrees2rad(angle)) ) 
                        sy = str( d*math.cos(degrees2rad(angle)) )
                        sz = str( 0 )
                    else:
                        sx = sources_pos[n*3  ]
                        sy = sources_pos[n*3+1]
                        sz = sources_pos[n*3+2]

                    csim.write( "{} = {} {} {}\n".format(dir_n+c[n], sx,sy,sz) )

            sub.check_call(['csim',path_csim])
            execute_ecoduet_and_bss_eval_combi(folder,test,c,dirs,N,duetcfg)
            
    else:
        print("Invalid mixer mode!")
        exit(1)




test(sys.argv[1]) # .test filepath





