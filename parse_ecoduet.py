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
        else:
            error("Unrecognized line precedes the original and estimated lines: <"+line+">")

    return (N, Ne, deg_o, deg_e, o, e)


if __name__ == "__main__":
    eco = parse_ecoduet("ecoduet.log") 
    p(eco)
