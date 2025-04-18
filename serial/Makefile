CXX      = g++
CXXFLAGS = -O3 -std=c++17

all: kmeans_serial

kmeans_serial: kmeans_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f kmeans_serial
