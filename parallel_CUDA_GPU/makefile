# Makefile for compiling kmeans3d.cu without optimization

# CUDA compiler
NVCC := nvcc

# Source file and output name
SRC := kmeans3d.cu
OUT := kmeans3d

# Target architecture (SM 6.0 for atomicAdd(double*))
ARCH := -arch=sm_60

# Compile rule
all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(ARCH) -o $(OUT) $(SRC)

# Clean rule
clean:
	rm -f $(OUT)

