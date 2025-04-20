#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    double x, y, z, minDist;
    int cluster;
} Point;

__global__ void distance(Point data[], Point centroids[], int itemNum){
    // thread id of current block (on x axis)
    int tid = threadIdx.x;

    int bid = blockIdx.x;
    int bx = blockDim.x;

    int gid = bx*bid + tid; 

    int k = 9;
    if(gid < itemNum) {
        for(int i = 0; i < k; i++) {
        	double dist = 
        	((data[gid].x - centroids[i].x) * (data[gid].x - centroids[i].x)) +
        	((data[gid].y - centroids[i].y) * (data[gid].y - centroids[i].y)) +
        	((data[gid].z - centroids[i].z) * (data[gid].z - centroids[i].z));
        	if(dist < data[gid].minDist) {
        		data[gid].minDist = dist;
        		data[gid].cluster = i;
        	}
        }
    }

}

extern "C" void launchDistance(Point data[], Point centroids[], int itemNum) {
    // Launch GPU kernel
    int k = 9;
    Point *gpuData;
    Point *gpuCentroids;


    //allocate memory
    cudaMalloc(&gpuData, itemNum * sizeof(point));
    cudaMalloc(&gpuCentroids, k * sizeof(point)); 

    cudaMemcpy(gpuData, data, itemNum * sizeof(point), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCentroids, centroids, k * sizeof(point), cudaMemcpyHostToDevice);

    int gpuBlockSize = 256;
    int gpuBlockNum = (itemNum + gpuBlockSize - 1) / gpuBlockSize;
   	distance<<<gpuBlockNum, gpuBlockSize>>>(gpuData, gpuCentroids, itemNum);

    // cuda synch barrier
    cudaDeviceSynchronize();

    cudaMemcpy(data, gpuData, itemNum * sizeof(point), cudaMemcpyDeviceToHost);

    cudaFree(gpuData);
    cudaFree(gpuCentroids);
}
