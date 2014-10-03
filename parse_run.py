#! /usr/bin/python3

from parse_ecoduet import *
from parse_bss_eval import *


def exec_bss_eval_static_and_ibm(bss_static_logpath, bss_ibm_logpath):
    """ Executes the BSS Eval toolkit in parallel for the current data. """

    t_static = thread_bss_eval_static(bss_static_logpath)
    t_ibm    = thread_bss_eval_ibm   (bss_ibm_logpath)
    t_static.start()
    t_ibm.start()
    t_static.join()
    t_ibm.join()

def exec_bss_eval_dynamic_and_ibm(bss_dynamic_logpath, bss_ibm_logpath):
    """ Executes the BSS Eval toolkit in parallel for the current data. """

    t_dynamic = thread_bss_eval_dynamic(bss_dynamic_logpath)
    t_ibm    = thread_bss_eval_ibm   (bss_ibm_logpath)
    t_dynamic.start()
    t_ibm.start()
    t_dynamic.join()
    t_ibm.join()


def parse_run(ecoduet_logpath, bss_eval_logpath):
    
    eco = parse_ecoduet(ecoduet_logpath)

    N = eco[0]
    Ne = eco[1]

    check(N==Ne, "Not implemented for Ne!=N")
    check((not eco[2]) and (not eco[3]), "ABORTING: Degeneracies occured!")

    bss = parse_bss_eval(bss_eval_logpath)

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

    if (eco_e): # If we're not using an ibm otherwise there are no estimates.
        for i_e in range(Ne):
            match = eco_e[i_e][0]
            check(match == bss_e[i_e][0], "LETHAL: BSS eval found a different permutation!")

            e.append(eco_e[i_e]+bss_e[i_e][1:])


    
    # eco == (N, Ne, deg_o , deg_e , o, e, SNR0)
    return   (N, Ne, eco[2], eco[3], o, e, eco[6])


if __name__ == "__main__":
    p( parse_run("ecoduet.log","bss_eval.log") )

