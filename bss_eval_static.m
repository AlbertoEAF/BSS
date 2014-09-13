function [SDR,SIR,SAR,perm SDRe,SIRe,SARe,perme] = bss_eval_static

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'x*_rebuilt.wav';

[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern, Spattern,1);

end
