function [SDR,SIR,SAR] = print_bss_eval_stats(SDR,SIR,SAR,perm)
% Uses bss_eval and sorts the results by perm so that sources match. It then prints
% the result.

%[SDR,SIR,SAR,perm] = bss_eval(Sepattern, Spattern);
original_perm = perm;

a = [SDR,SIR,SAR,perm];
a = sortrows(a,4);

N = length(SDR);

fprintf('SDR (dB)\t&\tSIR (dB)\t&\tSAR (dB)  \\\\\n');
for n = 1:N
    fprintf('%.3f\t\t&\t%.3f\t\t&\t%.3f\t  \\\\\n', SDR(n), SIR(n), SAR(n));
end

fprintf('\nEstimated sources were assigned to the true sources: ');
for n = 1:N
    fprintf('%d ', original_perm(n));
end
fprintf('\n');

end
