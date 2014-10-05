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

    
    for i in range(len(ecologs)):
        ecolog  = folder+"/"+ecologs [i]
        ecologi = folder+"/"+ecologis[i]
        bsslog  = folder+"/"+bsslogs [i]
        bsslogi = folder+"/"+bsslogis[i]
    
        print(GREEN, " #####", NOCOLOR)
        os.system("cat {} {} {} {}".format(ecolog,ecologi,bsslog,bsslogi))

        # (N, Ne, deg_o , deg_e , o, e, SNR0)
        N , Ne , deg_o , deg_e , o  , e , _    = parse_run(ecolog , bsslog , 0)
        _ , _  , _     , _     , oi , _ , SNR0 = parse_run(ecologi, bsslogi, 0)


if __name__ == "__main__":
    parse_test(sys.argv[1])
