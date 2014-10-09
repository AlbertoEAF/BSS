#! /usr/bin/python3

from math import sqrt 

from parse_run import *
from ConfigParser import *

from color_codes import *
import os
import sys

import numpy as np

MAX_DEGENERACIES = 2 # Above this it sums everything up

def parse_test(testpath):
    test = ConfigParser(testpath,"parent_localpath")

    

    folder = os.path.split(os.path.abspath(testpath))[0]

    files = [ f for f in os.listdir(folder) ] # if os.path.isfile(f) ]

    ecologs  = [ f for f in files if f[-7:] == ".ecolog"  ]
    ecologis = [ f for f in files if f[-8:] == ".ecologi" ]
    bsslogs  = [ f for f in files if f[-7:] == ".bsslog"  ]
    bsslogis = [ f for f in files if f[-8:] == ".bsslogi" ]

    disabled_bss_eval = False
    if test.i("disable_bss_eval"):
        disabled_bss_eval = True
        print(RED, "PARSER IS RUNNING WITH DISABLED BSS EVAL. Press [ENTER] to continue...", NOCOLOR, sep="", end="", flush=True)
        input("")
    elif not (len(ecologs)==len(ecologis)==len(bsslogs)==len(bsslogis)):
        p(ecologs)
        p(ecologis)
        p(bsslogs)
        p(bsslogis)

        error("Not all the .eco and .bss logfiles exist. Check that you haven't disabled bss_eval in the test. Also note that programming errors in the MATLAB BSS toolkit that crash the script will pass as OK. Run manually and if it reaches the end of the program it's not from that.")

    total_tests = len(ecologs)

    pass_count = 0
    

    targetvalues  = np.zeros( (len(ecologs),4) ) #len entries of (SNR,SDR,SIR,SAR)
    # ideal values
    targetvaluesi = np.zeros( (len(ecologs),4) ) #len entries of (SNR,SDR,SIR,SAR)
    snr0s         = np.zeros( len(ecologs) )

    eNpositive    = np.zeros( len(ecologs) )
    eNnegative    = np.zeros( len(ecologs) )
    deg_intrinsic = 0

    # To histogram of degeneracy per each simulation. the last bin is for results of max_degeneracies or higher
    eNpos_hist = np.zeros( MAX_DEGENERACIES+1 )
    eNneg_hist = np.zeros( MAX_DEGENERACIES+1 )

    for i in range(len(ecologs)):
        # FS sorting doesn't work in the same way for all unfortunately-> We must regenerate the names ourselves.
        combi_name = ecologs[i][:-len(".ecolog")]
        ecolog  = folder+"/"+combi_name+".ecolog"
        ecologi = folder+"/"+combi_name+".ecologi"
        bsslog  = folder+"/"+combi_name+".bsslog"
        bsslogi = folder+"/"+combi_name+".bsslogi"


        print(GREEN,os.path.split(ecolog)[1][:-len(".ecolog")],NOCOLOR, sep="", end=" ", flush=True)
        
        if disabled_bss_eval:
            (N, Ne, deg_o, deg_e, o, e, SNR0) = parse_ecoduet(ecolog)
        else:
            N,Ne,deg_o,deg_e,o,e,SNR0,oi = parse_test_logs(ecolog,bsslog,ecologi,bsslogi)
        # Error in number of estimated sources.
        eN = Ne-N 
                
        if (eN):
            print("{} {} deg({},{})".format(N,Ne,deg_o,deg_e))
        else:
            print("")

        # Degeneracies origin checking. Standard degeneracies occur just because Ne!=N but intrinsic ones happen when one source wasn't well separated. For instance with N=2 there was Ne=2 but the 2 peaks more closely match one of the sources.
        if eN:
            if eN > 0:
                eNpositive[pass_count] = eN
                eNpos_hist[eN] += 1
                check (deg_o ==   0 and deg_e == eN, "Intrinsic Degeneracies!")
            else:
                eNnegative[pass_count] = -eN
                eNneg_hist[-eN] += 1
                check (deg_o == -eN and deg_e == 0 , "Intrinsic degeneracies!")
        else:
            if (deg_o or deg_e):
                deg_intrinsic += 1
                print(RED,"------------------------------>Deg:",deg_o,deg_e, NOCOLOR)
                if test.i("check_degeneracy"):
                    check(deg_o == 0 and deg_e == 0, "Intrinsic Degeneracies")
            else:
                eNpos_hist[0] += 1
                eNneg_hist[0] += 1
        

        # parse_run's o = (match, Dtotal, Dtotal_per_sample, SNR, SDR, SIR, SAR)

        
        if not deg_o and not deg_e:
            if not disabled_bss_eval:
                targetvalues[pass_count]  = (o[0][3] ,o[0][4] ,o[0][5] ,o[0][6])
                targetvaluesi[pass_count] = (oi[0][3],oi[0][4],oi[0][5],oi[0][6])
            #targetSNR = o[0][3]
            #targetSDR = o[0][4]
            #targetSIR = o[0][5]
            #targetSAR = o[0][6]
            pass_count += 1 # Compact all entries to slice the end (we no longer average results where degeneracies occurred)



    if disabled_bss_eval:
        print ("No degeneracies in ", pass_count, "/", len(ecologs))
        exit

        #        print(N,Ne,deg_o,deg_e,o,e,SNR0,oi)

        #        print(targetSNR,targetSDR,targetSIR,targetSAR)
        

    eNpos_avg = eNpositive[:pass_count].sum()/float(len(ecologs))
    eNpos_std = sqrt ((eNpositive[:pass_count]**2).sum()/float(len(ecologs)-1))
    eNneg_avg = eNnegative[:pass_count].sum()/float(len(ecologs))
    eNneg_std = sqrt ((eNnegative[:pass_count]**2).sum()/float(len(ecologs)-1))

    print(eNpos_avg, eNpos_std, eNneg_avg, eNneg_std)

    if not pass_count:
        error("No tests passed")

    targetavgratios    = targetvalues[:pass_count].mean(axis=0) #(SNR,SDR,SIR,SAR)
    targetstddevratios = targetvalues[:pass_count].std(axis=0, ddof=1, dtype=np.float64)

    targetavgratiosi    = targetvaluesi[:pass_count].mean(axis=0) #(SNR,SDR,SIR,SAR)
    targetstddevratiosi = targetvaluesi[:pass_count].std(axis=0, ddof=1, dtype=np.float64)

    print("deg+", eNpos_hist)
    print("deg-", eNneg_hist)

    print("")

    print("SNR (dB)\tSDR (dB)\tSIR (dB)\tSAR (dB)")
    for i in range(4):
        print("{:.3}+-{:.3}\t".format(targetavgratios[i],targetstddevratios[i]), end="")
    print("")

    print("iSNR (dB)\tiSDR (dB)\tiSIR (dB)\tiSAR (dB)")
    for i in range(4):
        print("{:.3}+-{:.3}\t".format(targetavgratiosi[i],targetstddevratiosi[i]), end="")
    print("")

    log = open(testpath[:-len(".test")]+".degeneracies",'w')
    #log.write("{} {} {} {}".format(eNpos_avg, eNpos_std, eNneg_avg, eNneg_std))
    for i in range(MAX_DEGENERACIES+1):
        log.write(str(eNpos_hist[i])+" ")
    log.write("\n")
    for i in range(MAX_DEGENERACIES+1):
        log.write(str(eNneg_hist[i])+" ")
    log.write("\n")
    log.close()

    log = open(testpath[:-len(".test")]+".results",'w')
    log.write("SNR (dB)\tSDR (dB)\tSIR (dB)\tSAR (dB)\n")
    for i in range(4):
        log.write("{:.3}+-{:.3}\t".format(targetavgratios[i],targetstddevratios[i]))
    log.write("\n")
    log.close()
        
    log = open(testpath[:-len(".test")]+".iresults",'w')
    log.write("iSNR (dB)\tiSDR (dB)\tiSIR (dB)\tiSAR (dB)\n")
    for i in range(4):
        log.write("{:.3}+-{:.3}\t".format(targetavgratiosi[i],targetstddevratiosi[i]))
    log.write("\n")
    log.close()

    


    #    print(SNR,SDR,SIR,SAR)


def parse_test_logs(ecolog,bsslog,ecologi,bsslogi):

    #os.system("cat {} {} {} {}".format(ecolog,ecologi,bsslog,bsslogi))

    # (N, Ne, deg_o , deg_e , o, e, SNR0)
    N , Ne , deg_o , deg_e , o  , e , _    = parse_run(ecolog , bsslog , 0)
    _ , _  , _     , _     , oi , _ , SNR0 = parse_run(ecologi, bsslogi, 0)

    return (N,Ne,deg_o,deg_e,o,e,SNR0,oi) 
    

if __name__ == "__main__":
    parse_test(sys.argv[1]) # .test file
