#! /usr/bin/python3

import os
import subprocess as sub

import threading

def p(arg): print("<",arg,">",sep="")

def error(msg): 
    print(msg)
    exit(1)

def check(test, msg):
    if not (test):
        error(msg)

DEVNULL = open(os.devnull, 'w')


# Synchronous calls (look below for async helpers).
def mcli_call_bss_eval_static(logpath):
    sub.check_call(["rm","-f",logpath])
    sub.check_call(["mcli", "-f", "bss_eval_static", "\'"+logpath+"\'"], stderr=DEVNULL, stdout=DEVNULL)


def mcli_call_bss_eval_dynamic(logpath):
    sub.check_call(["rm","-f",logpath])
    sub.check_call(["mcli", "-f", "bss_eval_dynamic", "\'"+logpath+"\'"], stderr=DEVNULL, stdout=DEVNULL)

def mcli_call_bss_eval_ibm(logpath):
    sub.check_call(["rm","-f",logpath])
    sub.check_call(["mcli", "-f", "bss_eval_ibm", "\'"+logpath+"\'"], stderr=DEVNULL, stdout=DEVNULL)


# Async helpers of the mcli_calls which take lots of time. Get the thread, start it and join it (remember to use differnt logpaths for simultaneous execution of course!).
def thread_bss_eval_static(logpath):
    return threading.Thread(name="static",target=mcli_call_bss_eval_static, args=(logpath,))

def thread_bss_eval_dynamic(logpath):
    return threading.Thread(name="dynamic",target=mcli_call_bss_eval_dynamic, args=(logpath,))

def thread_bss_eval_ibm(logpath):
    return threading.Thread(name="ibm",target=mcli_call_bss_eval_ibm, args=(logpath,))



def parse_bss_eval(logpath):
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
            # (match, SDR,SIR,SAR)
            values = (int(l[0])-1, float(l[2]),float(l[5]),float(l[8])) 
            if line_is == ORIGINAL:
                o.append(values)
            else:
                e.append(values)
        else:
            error("Unrecognized line precedes the original and estimated lines: <"+line+">")


    return (o, e)

if __name__ == "__main__":
    mcli_call_bss_eval_static("bss_eval.log")
    bss = parse_bss_eval("bss_eval.log")
    p(bss)
    
