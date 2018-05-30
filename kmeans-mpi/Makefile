NVCC        = nvcc
MPIC++      = mpic++
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = kmeans
OBJ	        = cuda.o mpi_first.o mpi_io.o 
OBJ1        = mpi_first.o
OBJ2		= mpi_io.o
default: $(EXE)

test-mode: default
mpi_io.o: mpi_io.cu kmeans.h
	$(NVCC)	-c -g -o $(OBJ2) mpi_io.cu 
cuda.o: cuda.cu kmeans.h error.h
	$(NVCC) -c -g -o $@ cuda.cu 
mpi_first.o: mpi_first.cpp kmeans.h 
	$(MPIC++) -c -o $(OBJ1) mpi_first.cpp 

$(EXE): $(OBJ)
	$(MPIC++) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm $(OBJ)








