function [SDR,SIR,SAR, SDRe,SIRe,SARe] = bss_eval_static

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'x*_rebuilt.wav';

% First run the bss_eval toolkit.
[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern, Spattern);

% Print the best match for each true source.
print_bss_eval_stats(SDR , SIR , SAR , perm );
% Print the best match for each estimated source.
print_bss_eval_stats(SDRe, SIRe, SARe, perme);

end
