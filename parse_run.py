#! /usr/bin/python3

from parse_ecoduet import *
from parse_bss_eval import *

def parse_run(ecoduet_logpath, bss_eval_logpath):
    
    
    eco = parse_ecoduet(ecoduet_logpath)

#    mcli_call_bss_eval_static(bss_eval_logpath)
    bss = parse_bss_eval_static(bss_eval_logpath)

    N = eco[0]
    Ne = eco[1]

    check(N==Ne, "Not implemented yet for Ne!=N")
    check((not eco[2]) and (not eco[3]), "ABORTING: Degeneracies occured!")

    eco_o = eco[4]
    eco_e = eco[5]

    bss_o = bss[0]
    bss_e = bss[1]

    # Merge the data from both files:
    #
    # ecoduet.log:
    #   (match, Dtotal,Dtotal_per_sample,SNR)
    # bss_eval.log:
    #   (match, SDR,SIR,SAR)
    #
    # into:
    #      (match, Dtotal, Dtotal_per_sample, SNR, SDR, SIR, SAR)
    o = []
    e = []

    for i_o in range(N):
        match = eco_o[i_o][0]
        check(match == bss_o[i_o][0], "LETHAL: BSS eval found a different permutation!")

        o.append(eco_o[i_o]+bss_o[i_o][1:])


    for i_e in range(Ne):
        match = eco_e[i_e][0]
        check(match == bss_e[i_e][0], "LETHAL: BSS eval found a different permutation!")

        e.append(eco_e[i_e]+bss_e[i_e][1:])

    return (o,e)


if __name__ == "__main__":
    (o,e) = parse_run("ecoduet.log","bss_eval.log")
    p(o)
    p(e)

