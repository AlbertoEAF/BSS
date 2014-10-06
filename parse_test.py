#! /usr/bin/python3

from parse_run import *
from ConfigParser import *

from color_codes import *
import os
import sys

import numpy as np

def parse_test(testpath):
    test = ConfigParser(testpath)

    folder = os.path.split(os.path.abspath(testpath))[0]

    files = [ f for f in os.listdir(folder) ] # if os.path.isfile(f) ]

    ecologs  = [ f for f in files if f[-7:] == ".ecolog"  ]
    ecologis = [ f for f in files if f[-8:] == ".ecologi" ]
    bsslogs  = [ f for f in files if f[-7:] == ".bsslog"  ]
    bsslogis = [ f for f in files if f[-8:] == ".bsslogi" ]

    if not (len(ecologs)==len(ecologis)==len(bsslogs)==len(bsslogis)):
        p(ecologs)
        p(ecologis)
        p(bsslogs)
        p(bsslogis)
        error("Not all the .eco and .bss logfiles exist. Check that you haven't disabled bss_eval in the test. Also note that programming errors in the MATLAB BSS toolkit that crash the script will pass as OK. Run manually and if it reaches the end of the program it's not from that.")


#    targetSNRs = 
 #   targetSDRs = 
  #  targetSIRs = 
   # targetSARs = 

    targetvalues = np.zeros( (len(ecologs),4) ) #len entries of (SNR,SDR,SIR,SAR)
    # ideal values
    targetvaluesi = np.zeros( (len(ecologs),4) ) #len entries of (SNR,SDR,SIR,SAR)
    snr0s        = np.zeros( len(ecologs) )

    for i in range(len(ecologs)):
        # FS sorting doesn't work in the same way for all unfortunately-> We must regenerate the names ourselves.
        combi_name = ecologs[i][:-len(".ecolog")]
        ecolog  = folder+"/"+combi_name+".ecolog"
        ecologi = folder+"/"+combi_name+".ecologi"
        bsslog  = folder+"/"+combi_name+".bsslog"
        bsslogi = folder+"/"+combi_name+".bsslogi"

        print(GREEN, "Testing: ", os.path.split(ecolog)[1], NOCOLOR, sep="")

    



        N,Ne,deg_o,deg_e,o,e,SNR0,oi = parse_test_logs(ecolog,bsslog,ecologi,bsslogi)

        #check(Ne>=N, "Ne<N")
  
        # parse_run's o = (match, Dtotal, Dtotal_per_sample, SNR, SDR, SIR, SAR)

        targetvalues[i] = (o[0][3],o[0][4],o[0][5],o[0][6])
        targetvaluesi[i] = (oi[0][3],oi[0][4],oi[0][5],oi[0][6])
        #targetSNR = o[0][3]
        #targetSDR = o[0][4]
        #targetSIR = o[0][5]
        #targetSAR = o[0][6]

        

        #        print(N,Ne,deg_o,deg_e,o,e,SNR0,oi)

#        print(targetSNR,targetSDR,targetSIR,targetSAR)
        
    targetavgratios    = targetvalues.mean(axis=0) #(SNR,SDR,SIR,SAR)
    targetstddevratios = targetvalues.std(axis=0, ddof=1, dtype=np.float64)

    targetavgratiosi    = targetvaluesi.mean(axis=0) #(SNR,SDR,SIR,SAR)
    targetstddevratiosi = targetvaluesi.std(axis=0, ddof=1, dtype=np.float64)

    print("SNR (dB)\tSDR (dB)\tSIR (dB)\tSAR (dB)")
    for i in range(4):
        print("{:.3}+-{:.3}\t".format(targetavgratios[i],targetstddevratios[i]), end="")
    print("")

    print("iSNR (dB)\tiSDR (dB)\tiSIR (dB)\tiSAR (dB)")
    for i in range(4):
        print("{:.3}+-{:.3}\t".format(targetavgratiosi[i],targetstddevratiosi[i]), end="")
    print("")

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
