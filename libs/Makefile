DEBUG = 


CC = g++
CCFLAGS = -O2 $(DEBUG) -Wuninitialized -Wextra -Winit-self

# Small Random Libs (SRL)
libSRL.a: String.o config_parser.o safedl.o 
	ar rcs libSRL.a String.o config_parser.o safedl.o

String.o: String.cpp String.h
	$(CC) $(CCFLAGS) -c String.cpp 

config_parser.o: config_parser.h config_parser.cpp
	$(CC) $(CCFLAGS) -c config_parser.cpp

safedl.o: safedl.h safedl.c
	$(CC) $(CCFLAGS) -c safedl.c


clean:
	rm -f *.a *.o *~
