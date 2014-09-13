function [SDR,SIR,SAR,perm] = bss_eval(Sepattern, Spattern)
% Calcs bss_eval for a pattern of input filenames. It is not restricted to
% Ne=N.

clc();

Sfiles  = dir(Spattern);
Sefiles = dir(Sepattern);

N  = length(Sfiles);
Ne = length(Sefiles);

if (N==0 || Ne==0)
    error('No sources or estimated sources found.');
end

% Real all the .wav files
cells_s  = cell(N ,1);
cells_se = cell(Ne,1);  

for n = 1:N
    cells_s{n} = wavread(strcat(fileparts(Spattern),'/',Sfiles(n).name));
end

for ne = 1:Ne
    cells_se{ne} = wavread(strcat(fileparts(Sepattern),'/',Sefiles(ne).name));
end   


% Find the number of samples
samples           = max(cellfun(@length, cells_s ));
estimated_samples = max(cellfun(@length, cells_se));

if estimated_samples ~= samples
    display('Program did not run for sure. Estimated samples should match the number of samples.');
    return;
end

% Consolidate the .wav irregularly-sized cells into matrices !!!SLOW!!!
% (Copies one datapoint at a time.)
s  = zeros(N , samples);
se = zeros(Ne, samples);

for n = 1:N
    for sample = 1:length(cells_s{n})
        s(n,sample) = cells_s{n}(sample);
    end
end

for n = 1:N
    for sample = 1:length(cells_se{n})
        se(n,sample) = cells_se{n}(sample);
    end
end

[SDR,SIR,SAR,perm] = bss_eval_sources_multi(se,s);


end % eof






