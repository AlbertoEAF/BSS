function [SDR,SIR,SAR] = bss_eval_static

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'x*_rebuilt.wav';

% First run the bss_eval toolkit.
[SDR,SIR,SAR,perm] = bss_eval(Sepattern, Spattern);
% Now sort the results and print.
[SDR,SIR,SAR]      = print_bss_eval_stats(SDR,SIR,SAR,perm);

end
