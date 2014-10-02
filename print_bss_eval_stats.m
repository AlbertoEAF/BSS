function print_bss_eval_stats(SDR,SIR,SAR,perm)
% Prints the bss_eval results in a table ready for latex.

N = length(SDR);

fprintf('match\t&\tSDR (dB)\t&\tSIR (dB)\t&\tSAR (dB)  \\\\\n');
for n = 1:N
    fprintf('%d\t&\t%.3f\t\t&\t%.3f\t\t&\t%.3f\t  \\\\\n', perm(n), SDR(n), SIR(n), SAR(n));
end


fprintf('\n\n');


end