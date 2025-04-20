all:
  mpicc -c kMeansMpi.cpp -o main.o
  nvcc -c kMeansCuda.cu -o cuda_main.o
  mpicc main.o cuda_main.o -lcudart -lstdc++ -llzma -o kMeans
