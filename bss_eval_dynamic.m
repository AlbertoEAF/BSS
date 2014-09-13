function [SDR,SIR,SAR,perm SDRe,SIRe,SARe,perme] = bss_eval_dynamic

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'xstream*_rebuilt.wav';

[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern, Spattern,1);

end
