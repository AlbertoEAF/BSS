CC=clang++


DEBUG= -g -G # -DNDEBUG
EXTRACCFLAGS =   $(DEBUG) -ffunction-sections -Wl,-gc-sections #-Wno-unused-variable #-Wno-unused-but-set-variable
OPTIMIZATION_FLAGS= -O2 # -DNDEBUG
OMP= -fopenmp
CCFLAGS= -Wall $(EXTRACCFLAGS) -std=c++11 $(OPTIMIZATION_FLAGS) $(OMP)


PRECISION= -DDOUBLE_PRECISION

## Libs
FFTLIBS= -lfftw3
WAVLIBS= -lsndfile
MATHLIBS= -lm
#ARMADILLO_LIB= -larmadillo #-fwhole-program # whole-program allows merging operations in armadillo

#Small Random Libs
LIBS_SRL = -Llibs -lSRL

LIBS = $(FFTLIBS) $(WAVLIBS) $(MATHLIBS) $(LIBS_SRL) $(ARMADILLO_LIB)


all: IdList duet ecoduet drvb

get_config_var: get_config_var.cpp
	$(CC) -o get_config_var get_config_var.cpp $(LIBS_SRL)

IdList: IdList.cpp IdList.h
	$(CC) $(CCFLAGS) -c IdList.cpp

drvb: libs
	$(CC) $(CCFLAGS) -o latedereverb latedereverb.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS) 	

csim: libs csim.cpp
	$(CC) $(CCFLAGS) -o csim csim.cpp $(WAVLIBS) $(LIBS_SRL) -std=c++11 gnuplot_ipp/gnuplot_ipp.o

duet: libs 
	$(CC) $(CCFLAGS) -o d duet.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS) 

OptionParser.o: OptionParser.cpp OptionParser.h
	$(CC) -c OptionParser.cpp -std=c++11

ecoduet: libs IdList OptionParser.o
	$(CC) $(CCFLAGS) -o r ecoduet.cpp gnuplot_ipp/gnuplot_ipp.o timer.o IdList.o OptionParser.o $(PRECISION) $(LIBS) 


duet.h.gch: duet.h Buffer.h Matrix.h Histogram2D.h array_ops.h types.h libs/config_parser.h wav.h gnuplot_ipp/gnuplot_ipp.h filters.h extra.h libs/timer.h RankList.h
	$(CC) -c duet.h -o duet.h.gch


fft_convolution: libs 
	$(CC) $(CCFLAGS) -o c fft_convolution.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)

utils: libs convolve.cpp invert_ir.cpp
	$(CC) $(CCFLAGS) -o convolve convolve.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)	
	$(CC) $(CCFLAGS) -o convolve2 convolve2.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)	
	$(CC) $(CCFLAGS) -o invert_ir invert_ir.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)
	$(CC) $(CCFLAGS) -o nlms nlms.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)
	$(CC) $(CCFLAGS) -o reverse reverse.cpp $(PRECISION) $(LIBS)
	$(CC) $(CCFLAGS) -o xcorr xcorr.cpp $(PRECISION) $(LIBS)
	$(CC) $(CCFLAGS) -o view view.cpp gnuplot_ipp/gnuplot_ipp.o $(PRECISION) $(LIBS)
	$(CC) $(CCFLAGS) -o spectrum spectrum.cpp gnuplot_ipp/gnuplot_ipp.o timer.o IdList.o $(PRECISION) $(LIBS) 

window_plots: window_plots.cpp
	$(CC) $(CCFLAGS) -o window_plots window_plots.cpp gnuplot_ipp/gnuplot_ipp.o timer.o IdList.o $(PRECISION) $(LIBS) 

ssub: IdList
	$(CC) $(CCFLAGS) -o ssub ssub.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS) 

u:
	g++ -DNDEBUG -O2 -o xcorr xcorr.cpp  $(LIBS) -fopenmp -std=c++11

convolution: libs 
	$(CC) $(CCFLAGS) -o t convolution_test.cpp gnuplot_ipp/gnuplot_ipp.o timer.o $(PRECISION) $(LIBS)

fft: libs
	$(CC) $(CCFLAGS) -o f fft.cpp gnuplot_ipp/gnuplot_ipp.o $(LIBS) timer.o $(PRECISION)


## JACK

sp: sim_player.cpp
	$(CC) $(CCFLAGS) -o sp sim_player.cpp -lm -ljack -lsndfile $(LIBS_SRL) 


## Other


test: libs test.cpp
	$(CC) $(CCFLAGS) -o ttt test.cpp gnuplot_ipp/gnuplot_ipp.o $(LIBS) $(PRECISION)

wav: 
	$(CC) -o t_write_wav write_wav.cpp $(LIBS) 


cleanhists:
	rm -f gnuplot_tmpdatafile_* hist_render2D/* hist_render3D/* hist_dats/*



### Libs

libs: gnuplotIpp SRL timer.o

gnuplotIpp: gnuplot_ipp/gnuplot_i.c gnuplot_ipp/gnuplot_i.h
	cd gnuplot_ipp && make cpp

SRL:
	cd libs; make

timer.o: libs/timer.cpp libs/timer.h
	$(CC) -c libs/timer.cpp



render:
	bash render.sh


src:
	rm -rf src
	mkdir src
	cp *.h src
	cp *.cpp src
	cp *.c src
	cp mix src
	cp run src
	cp gnuplot_ipp src
	cp libs src
	cp render_cfg src
	cp Makefile src
