DEBUG = 

CC = clang++
CCFLAGS = -O2 $(DEBUG) -I`pwd`/.. -I`pwd`


#Using the non-error-prone chaining B-type. Change to A-type for smaller binaries at the end of the development cycle. Check the how_to.

libBuffers.a: Buffer.o Buffers.o BufferPool.o Matrix.o
	ar rcs libBuffers.a Buffer.o Buffers.o BufferPool.o Matrix.o

Buffer.o: Buffer.cpp Buffer.h BufferDeclaration.h  BufferInstantiations.h
	$(CC) $(CCFLAGS) -c Buffer.cpp 

Buffers.o: Buffer.o Buffers.cpp Buffers.h BuffersDeclaration.h BuffersInstantiations.h 
	$(CC) $(CCFLAGS) -c Buffers.cpp 

BufferPool.o: Buffer.o BufferPool.cpp BufferPool.h BufferPoolDeclaration.h BufferPoolInstantiations.h
	$(CC) $(CCFLAGS) -c BufferPool.cpp -std=c++11 # uses std::mutex (C++11)

Matrix.o: Matrix.cpp Matrix.h MatrixDeclaration.h MatrixInstantiations.h
	$(CC) $(CCFLAGS) -c Matrix.cpp

clean:
	rm -f *.a *.o *~ 
