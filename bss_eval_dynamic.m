function [SDR,SIR,SAR,perm SDRe,SIRe,SARe,perme] = bss_eval_dynamic(logfile)

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'xstream*_rebuilt.wav';

[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern, Spattern,1,logfile);


% Print the best match for each true source.
print_bss_eval_stats(SDR , SIR , SAR , perm );
% Print the best match for each estimated source.
print_bss_eval_stats(SDRe, SIRe, SARe, perme);


end
