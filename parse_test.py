#! /usr/bin/python3

from parse_run import *
from ConfigParser import *

from color_codes import *
import os
import sys

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

    for i in range(len(ecologs)):
        # FS sorting doesn't work in the same way for all unfortunately-> We must regenerate the names ourselves.
        combi_name = ecologs[i][:-len(".ecolog")]
        ecolog  = folder+"/"+combi_name+".ecolog"
        ecologi = folder+"/"+combi_name+".ecologi"
        bsslog  = folder+"/"+combi_name+".bsslog"
        bsslogi = folder+"/"+combi_name+".bsslogi"

        print(GREEN, "Testing: ", os.path.split(ecolog)[1], NOCOLOR, sep="")

    



        N,Ne,deg_o,deg_e,o,e,SNR0,oi = parse_test_logs(ecolog,bsslog,ecologi,bsslogi)

  
        print(N,Ne,deg_o,deg_e,o,e,SNR0,oi)

        

def parse_test_logs(ecolog,bsslog,ecologi,bsslogi):

    os.system("cat {} {} {} {}".format(ecolog,ecologi,bsslog,bsslogi))

    # (N, Ne, deg_o , deg_e , o, e, SNR0)
    N , Ne , deg_o , deg_e , o  , e , _    = parse_run(ecolog , bsslog , 0)
    _ , _  , _     , _     , oi , _ , SNR0 = parse_run(ecologi, bsslogi, 0)

    return (N,Ne,deg_o,deg_e,o,e,SNR0,oi) 
    

if __name__ == "__main__":
    parse_test(sys.argv[1])
