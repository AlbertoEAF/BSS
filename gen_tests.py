#! /usr/bin/python3

### usage: prgm <test_file.test>

from os import listdir
from os.path import isfile, join
import os
import subprocess as sub
import itertools
import re

from ConfigParser import *
from color_codes import *
from parse_run import *


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

    test = ConfigParser(test_file)

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



def test(test_file):
    """
    Run all tests and save the logs for later processing.

    ecoduet saves 2 logs: ecoduet.log and ecoduet_ibm.log
    similarly we'll call bss_eval to output: bss_eval.log and bss_eval_ibm.log
    """
    
    folder = test_file[:test_file.rfind("/")+1]

    test = ConfigParser(test_file)
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

    if test['mixer'] == 'mix':
        for i_c in range(len(combinations)):
            c = combinations[i_c]
            print( GREEN, "Testing({}/{}): {}".format(i_c+1,len(combinations),c) , NOCOLOR, sep="")

            sub.check_call(['mix']+[ dirs[n]+'/'+c[n] for n in range(N) ])

            combi_name = "_".join([ c[n][:-4] for n in range(N) ])

            ecolog  = folder+combi_name+".ecolog"
            ecologi = folder+combi_name+".ecologi" # ibm

            bsslog  = folder+combi_name+".bsslog"
            bsslogi = folder+combi_name+".bsslogi" # ibm

            print("Running ecoduet...",end="",flush=True)



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



    elif test['mixer'] == 'csim':
        print("Not implemented!")
        exit(1)
    else:
        print("Invalid mixer mode!")
        exit(1)




test(sys.argv[1]) # .test filepath





