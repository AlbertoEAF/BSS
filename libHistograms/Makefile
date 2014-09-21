DEBUG = 


CC = clang++
CCFLAGS = -O2 $(DEBUG) -I`pwd`/..


# Small Random Libs (libHistograms)
libHistograms.a: Histogram.o Histogram2D.o
	echo $(CCFLAGS)
	ar rcs libHistograms.a Histogram.o Histogram2D.o

Histogram.o: Histogram.cpp Histogram.h HistogramDeclaration.h 
	$(CC) $(CCFLAGS) -c Histogram.cpp 

Histogram2D.o: Histogram2D.cpp Histogram2D.h Histogram2DDeclaration.h
	$(CC) $(CCFLAGS) -c Histogram2D.cpp


clean:
	rm -f *.a *.o *~ 
