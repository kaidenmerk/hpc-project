CC       = g++
CXXFLAGS = -O3 -fopenmp -std=c++11
SRC      = kmeans_cpu_shared.cpp
TARGET   = kmeans_cpu_shared

# Default target builds the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CXXFLAGS) $< -o $@

# Clean up the binary
clean:
	rm -f $(TARGET)

.PHONY: all clean
