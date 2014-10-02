#! /usr/bin/python3

from os import listdir
from os.path import isfile, join

import os
import subprocess as sub


import re
from ConfigParser import *

from color_codes import *

from parse_run import *

def raw_string(s):
    if isinstance(s, str):
        s = s.encode('string-escape')
    elif isinstance(s, unicode):
        s = s.encode('unicode-escape')
    return s


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


    sources = test['sources'].split()

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

    combinations = []

    import itertools

    combs_count = 0
    excls_count = 0

    all_combinations = itertools.product(*files)

    for combination in all_combinations:
        combs_count += 1

        exclude_flag = False
        for n1 in range(N-1):
            for n2 in range(1,N):
                if exclude(combination[n1], combination[n2], rules):
                    exclude_flag = True

        if exclude_flag:
            excls_count += 1
        else:
            combinations.append(combination)


    return combinations



def tests(test_file):
    """
    Run all tests and save the logs for later processing.

    ecoduet saves 2 logs: ecoduet.log and ecoduet_ibm.log
    similarly we'll call bss_eval to output: bss_eval.log and bss_eval_ibm.log
    """
    
    folder = test_file[:test_file.rfind("/")]

    test = ConfigParser(test_file)
    combinations = gen_combinations(test_file)

    if test['mixer'] == 'mix':
        for i_c in range(len(combinations)):
            c = combinations[i_c]
            print("Testing (",i_c,"/",len(combinations),") : ", c, sep="")

            sub.check_call(['mix']+[ dirs[n]+'/'+c[n] for n in range(N) ])

            sub.check_call(["rm","-f","ecoduet.log","bss_eval.log","bss_eval_ibm.log"])

            out=sub.check_output(['r','omni.cfg'])

            (o,e) = parse_run("ecoduet.log", "bss_eval.log")

            print(RED,o,e,NOCOLOR)

    elif test['mixer'] == 'csim':
        print("Not implemented!")
        exit(1)
    else:
        print("Invalid mixer mode!")
        exit(1)




gen_test('t.test')


