#! /usr/bin/python3

import sys
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


def parse_ecoduet(logpath):
    """
    Parses the ecoduet log and returns N,Ne and the original and estimated lists with entries:
       [ (match_index Dtotal Dtotal/sample SNR), ... ] sorted by the original/estimated source index.
    """

    log = open(logpath, 'r')
    lines = [ line.strip() for line in log.readlines() if line.strip() ]
    log.close()

    o = []
    e = []

    N  = int(lines[0].split()[2])
    Ne = int(lines[0].split()[5])

    # State machine
    NONE = 0
    ORIGINAL = 1
    ESTIMATED = 2
    # state
    line_is = NONE

    SNR0 = []
    deg_e = 0

    for line in lines[1:]:
        if line[0] == "e": # line starts with e_match
            line_is = ORIGINAL
            continue
        elif line[0] == "o": # line starts with o_match
            line_is = ESTIMATED
            continue
        elif line_is != NONE: # If the file obeys the specs at least we must be in some state by now.

            # Parse degeneracies line.
            if line[0] == "#":
                if line_is == ORIGINAL:
                    deg_o = int(line.split("=")[1])
                else:
                    deg_e = int(line.split("=")[1])
                continue

            # Parse normal values line.
            l = line.split()
            # (match, Dtotal,Dtotal_per_sample,SNR)
            values = (int(l[0])-1, float(l[1]), float(l[2]), float(l[3])) 

            if line_is == ORIGINAL:
                o.append(values)
            else:
                e.append(values)
        else: # Line that precedes the standard results shows the SNR0 values.
            SNR0 = line.split()

    return (N, Ne, deg_o, deg_e, o, e, SNR0)


if __name__ == "__main__":
    if (len(sys.argv) == 1):
        logpath = input("Ecolog:")
    else:
        logpath = sys.argv[1]
    eco = parse_ecoduet(logpath) 
    print(eco)
