DEBUG = 

CC = clang++
CCFLAGS = -O2 $(DEBUG) -I${CURDIR}/../libBuffers -I${CURDIR}/..

libHistograms.a: Histogram.o Histogram2D.o
	ar rcs libHistograms.a Histogram.o Histogram2D.o

Histogram.o: Histogram.cpp Histogram.h HistogramDeclaration.h HistogramInstantiations.h
	$(CC) $(CCFLAGS) -c Histogram.cpp 

Histogram2D.o: Histogram2D.cpp Histogram2D.h Histogram2DDeclaration.h Histogram2DInstantiations.h
	$(CC) $(CCFLAGS) -c Histogram2D.cpp


clean:
	rm -f *.a *.o *~ 
