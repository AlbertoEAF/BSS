#! /usr/bin/python2.7

########## User configuration ##########

sources_folder = "sounds_database/"
output_folder = "sounds/"

# dictionary sources[wavfile] = (posx,posy,posz)
sources = {} 
sources["mike.wav"] = (.5,.1, 0.)
sources["mike2.wav"] = (-1.5,.1, 0.)

Delta_stretch = 1.

c = 340. # speed of sound (m/s)

#########################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys # sys.exit

source_files = [ sources_folder+source for source in sources.keys() ]
source_pos   = [ sources[source] for source in sources.keys() ]

def save_waves(data, out_prefix, rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    for i in range(data.shape[0]):
        scipy.io.wavfile.write(out_prefix+str(i)+".wav", rate, scaled[i])

def avg(data):
    return np.sum(data) / float(data.size)

def distance(a,b):
    return np.linalg.norm(np.asarray(b)-np.asarray(a))

def linear_interpolation(s, t, dt_s):
    """ Returns the linear interpolation of a signal with frame time = dt_s. """
    n = int(t/dt_s)
    t_n = n*dt_s

    if n >= s.size-1:
        return s[-1]

    return s[n] + (t-t_n)*(s[n+1]-s[n])/dt_s

def drop_interpolation(s, t, dt_s):
    """ Returns the drop interpolation of a signal with frame time = dt_s. """
    n = int(t/dt_s)

    return s[n]


def signal_linear_interpolation(signal, t, dt_s):
    output = np.zeros(signal.size)

    return output

def MaxNormalizeRows(array):
    return np.asarray(array / np.asmatrix(np.max(array, axis=1)).T)


######## CODE

def npify(array):
    return np.asarray([ np.asarray(element) for element in array ])

# load audio files
wav_data = []
sample_rate = None
for f in source_files:
    sr, data = scipy.io.wavfile.read(f)
    if len(data.shape) != 1:
        print "File {0} is not mono!".format(f)
        sys.exit(1)
    if sample_rate is None:
        sample_rate = sr
    else:
        assert(sample_rate == sr)
    wav_data.append(data)
# Put them all with the same size
L = np.max([ wav_source.size for wav_source in wav_data ])    
for i in range(len(wav_data)):
    wav_data[i] = np.pad(wav_data[i], (0,L-wav_data[i].size), "constant")
wav_data = np.asarray(wav_data).astype(float)


# Audio engineer's device configuration
print "Sample rate = {0} (Hz)".format(sample_rate)
f_max = sample_rate
w_max = 2*np.pi*f_max
Delta = np.pi * c / w_max * float(Delta_stretch)
print "Delta = {0} (cm)".format(Delta*100)
print "Delta/c = {0} (s)".format(Delta/c)
#mic_pos = [(0.,0.,0.), (0., Delta, 0.)] # DUET (2 mics)
mic_pos = [(-Delta/2.,0.,0.), (Delta/2., 0., 0.)] # DUET (2 mics)

N = len(sources)
M = len(mic_pos)


waves = MaxNormalizeRows( (np.c_[wav_data]).astype(float) )




D = np.zeros((M,N)) # delay matrix
A = np.zeros((M,N)) # attenuation matrix


for n in range(N):    
    for m in range(M):
        d_nm = distance(source_pos[n],mic_pos[m])
        D[m,n] = d_nm / c
        A[m,n] = 1 / (4*np.pi*d_nm*d_nm)

# We use relative coefficients for the DUET model in order to m=0
d = np.zeros((M-1,N)) 
a = np.zeros((M-1,N))

for m in range(1,M):
    d[m-1] = D[m] - D[0]
    a[m-1] = A[m] / A[0]

a = a - 1/a # a -> alpha transformation

print source_pos

print "D:\n", D
print "A:\n", A

print "d:\n", d
print "alpha:\n", a

print "(simulation error: port to C (for speed and precision) and use long double) d_n0: ", (distance(source_pos[0],mic_pos[1])-distance(source_pos[0],mic_pos[0]))/c


dt_s = 1/float(sample_rate)


# NOTE: we didn't extend the out_wave size!!! DO SO!
extra_samples = 0# int(3 / dt_s) # extra 3s of audio
out_waves = np.zeros((M, waves.shape[1]+extra_samples))


print "M =", M
print "N =", N

save_waves(waves, output_folder+"s", sample_rate)

# Generate output for each individual source contribution to each mic
for n in range(N):
    print "Generating microphone measurements for the source {0}/{1} ...".format(n+1,N)
    for i in range(out_waves.shape[1]):
        t_i = i*dt_s # interpolation access outside!!!
        for m in range(M):
            out_waves[m][i] = A[m,n] * linear_interpolation(waves[n],t_i-D[m,n], dt_s)

    save_waves(out_waves, output_folder + str(n)+"x", sample_rate)

# Generate output for the full mix (linear superposition)
print "Generating measurements for the full mix ..."
for i in range(out_waves.shape[1]):
    t_i = i*dt_s # interpolation access outside!!!
    for m in range(M):
        for n in range(N):
            out_waves[m][i] += A[m,n] * linear_interpolation(waves[n],t_i-D[m,n], dt_s)

save_waves(out_waves, output_folder + "x", sample_rate)

with open("simulation.log","wt") as log:
    log.write(str(N)+"\n")
    for n in range(N):
        log.write("{0} {1}\n".format(a[0][n], d[0][n]))

print "\n[DONE]\n"
