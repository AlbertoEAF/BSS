function [SDR,SIR,SAR,perm SDRe,SIRe,SARe,perme] = bss_eval_static (logfile)

Spattern  = 'sounds/s*x0.wav';
Sepattern = 'x*_rebuilt.wav';


[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern, Spattern);

diary on

diary(logfile);

% Print the best estimated source match for each true source.
fprintf('e_'); print_bss_eval_stats(SDR , SIR , SAR , perm );
% Print the best original match for each estimated source.
fprintf('o_'); print_bss_eval_stats(SDRe, SIRe, SARe, perme);



end

