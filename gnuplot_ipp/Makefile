# Choose the compilation target to save compilation time: c or cpp if you're using C or C++ respectively.

all: c cpp

c: gnuplot_i.o
cpp: gnuplot_ipp.o

gnuplot_i.o: gnuplot_i.c gnuplot_i.h
	gcc -c gnuplot_i.c -Wall

gnuplot_ipp.o: gnuplot_i.c gnuplot_i.h gnuplot_ipp.h
	g++ -c gnuplot_i.c -o gnuplot_ipp.o -Wall

clean:
	rm -f gnuplot_i.o gnuplot_ipp.o


src: clean
	rm -f gnuplot_ipp.zip *~
	zip -r gnuplot_ipp *