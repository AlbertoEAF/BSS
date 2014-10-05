function [SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval(Sepattern,Spattern, skip_time)
% Frontend utility to call bss_eval_source(_multi) from a pattern of
% filepaths. Since it uses bss_eval_source_multi Ne and N needn't be equal.

if nargin < 3, skip_time=0; end

Sfiles  = dir(Spattern);
Sefiles = dir(Sepattern);

N  = length(Sfiles);
Ne = length(Sefiles);

if (N==0 || Ne==0)
    error('Either true or estimated sources were not found in the filesystem.');
end

% Real all the .wav files
cells_s  = cell(N ,1);
cells_se = cell(Ne,1);  

for n = 1:N
    [cells_s{n},samplerate,~] = readwav(strcat(fileparts(Spattern),'/',Sfiles(n).name));
end

for ne = 1:Ne
    cells_se{ne} = readwav(strcat(fileparts(Sepattern),'/',Sefiles(ne).name));
end   

Ts = 1/double(samplerate);

% Coerce to double (doesn't work for strings)
skip_time=double(skip_time);

skip_samples = int64(skip_time/Ts);

% Find the number of samples
samples           = max(cellfun(@length, cells_s ));
estimated_samples = max(cellfun(@length, cells_se));

if estimated_samples ~= samples
    display('Program did not run for sure. Estimated samples should match the number of samples.');
    return;
end

% Consolidate the .wav irregularly-sized cells into matrices ( !!!SLOW!!! Copies one datapoint at a time. -> There's surely a better way)
s  = zeros(N , samples-skip_samples);
se = zeros(Ne, samples-skip_samples);

for n = 1:N
    for sample = 1+skip_samples:length(cells_s{n})
        s(n,sample) = cells_s{n}(sample);
    end
end

for n = 1:Ne
    for sample = 1+skip_samples:length(cells_se{n})
        se(n,sample) = cells_se{n}(sample);
    end
end

[SDR,SIR,SAR,perm, SDRe,SIRe,SARe,perme] = bss_eval_sources_multi(se,s);


end % eof






