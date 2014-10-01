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
    Parses the logfile and returns the original and estimated lists with entries:
    [ (matching_source_index,SDR,SIR,SAR) , ... ] sorted by the original/estimated source index.
    """

    diary_file = open(logpath, 'r')
    diary_lines = diary_file.readlines()
    diary_file.close()

    lines = [ line.strip() for line in diary_lines if line.strip() ]

    o = []
    e = []

    # State machine
    NONE = 0
    ORIGINAL = 1
    ESTIMATED = 2
    # state
    line_is = NONE

    for line in lines:
        if line[:7] == "e_match":
            line_is = ORIGINAL
            continue
        elif line[:7] == "o_match":
            line_is = ESTIMATED
            continue
        elif line_is != NONE: # If the file obeys the specs at least we must be in some state by now.
            l = line.split("\t")
            values = (int(l[0])-1, float(l[2]),float(l[5]),float(l[8])) # (match, SDR,SIR,SAR)
            if line_is == ORIGINAL:
                o.append(values)
            else:
                e.append(values)
        else:
            error("Unrecognized line precedes the original and estimated lines: <"+line+">")


    return (o, e)


logpath="staticMLABlog"
mcli_call_bss_eval_static(logpath)
dicts = parse_bss_eval_static(logpath)


p(dicts)
