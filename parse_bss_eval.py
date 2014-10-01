#! /usr/bin/python3

import os
import subprocess as sub

def p(arg): print("<",arg,">",sep="")

def error(msg): 
    print(msg)
    exit(1)

def check(test, msg):
    if not (test):
        error(msg)

DEVNULL = open(os.devnull, 'w')

def mcli_call_bss_eval_static(logpath):
    sub.check_call(["rm","-f",logpath])
    sub.check_call(["mcli", "bss_eval_static", "\'"+logpath+"\'"], stderr=DEVNULL, stdout=DEVNULL)


def parse_bss_eval_static(logpath):
    """
    Parses the logfile and returns the original and estimated dictionaries with entries:
    { matching_source_index : (SDR,SIR,SAR) , ... }.
    
    The original dictionary has each key as the index (starting at 1 !) of the matching peak.

    The estimated dictionary has as each match the index of the original source sorted on the filesystem by number (s0, s1) -> (1,2) indexes.
    """

    diary_file = open(logpath, 'r')
    diary_lines = diary_file.readlines()
    diary_file.close()

    lines = [ line.strip() for line in diary_lines if line.strip() ]

    dict_o = {}
    dict_e = {}


    # State machine
    # states
    NONE = 0
    ORIGINAL = 1
    ESTIMATED = 2
    # state
    line_is = NONE

    for line in lines:
        if line[:3] == "o_s":
            line_is = ORIGINAL
            continue
        elif line[:3] == "e_s":
            line_is = ESTIMATED
            continue
        elif line_is != NONE: # Already in some state
            l = line.split("\t")
            ratios = (float(l[2]),float(l[5]),float(l[8])) # (SDR,SIR,SAR)
            n = int(l[0])
            if line_is == ORIGINAL:
                check(n not in dict_o.keys(), "Degenerate solution.")
                dict_o[n] = ratios
            else:
                check(n not in dict_e.keys(), "Degenerate solution.")
                dict_e[n] = ratios

    return (dict_o, dict_e)


logpath="staticMLABlog"
mcli_call_bss_eval_static(logpath)
dicts = parse_bss_eval_static(logpath)
