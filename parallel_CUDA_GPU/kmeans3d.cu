// kmeans3d.cu
// CUDA implementation of k-means clustering on 3D PCA-reduced Spotify data
// Includes scaling study with different CUDA block sizes and deterministic clustering output

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#define N_CLUSTERS 9
#define MAX_ITERS 100
#define EPSILON 1e-6

// Kernel to assign each point to the nearest centroid using Euclidean distance in 3D
__global__ void assign_clusters(const double *x, const double *y, const double *z,
                                const double *cx, const double *cy, const double *cz,
                                int *labels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double min_dist = 1e20; // initialize with large distance
        int best = 0;
        for (int k = 0; k < N_CLUSTERS; ++k) {
            double dx = x[i] - cx[k];
            double dy = y[i] - cy[k];
            double dz = z[i] - cz[k];
            double dist = dx * dx + dy * dy + dz * dz;
            if (dist < min_dist) {
                min_dist = dist;
                best = k;
            }
        }
        labels[i] = best;
    }
}

// Kernel to reset centroid accumulators and counters before accumulation step
__global__ void reset_centroids(double *sum_x, double *sum_y, double *sum_z, int *count) {
    int k = threadIdx.x;
    if (k < N_CLUSTERS) {
        sum_x[k] = 0.0;
        sum_y[k] = 0.0;
        sum_z[k] = 0.0;
        count[k] = 0;
    }
}

// Kernel to accumulate sums for centroid updates using atomicAdd (requires sm_60+)
__global__ void accumulate_centroids(const double *x, const double *y, const double *z, const int *labels,
                                     double *sum_x, double *sum_y, double *sum_z, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int k = labels[i];
        atomicAdd(&sum_x[k], x[i]);
        atomicAdd(&sum_y[k], y[i]);
        atomicAdd(&sum_z[k], z[i]);
        atomicAdd(&count[k], 1);
    }
}

// Load PCA-reduced 3D data from CSV file
bool load_data(const std::string &filename,
               std::vector<double> &x, std::vector<double> &y, std::vector<double> &z) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sx, sy, sz;
        std::getline(ss, sx, ',');
        std::getline(ss, sy, ',');
        std::getline(ss, sz, ',');

        double xf = std::stod(sx);
        double yf = std::stod(sy);
        double zf = std::stod(sz);

        x.push_back(xf);
        y.push_back(yf);
        z.push_back(zf);
    }
    return true;
}

int main() {
    srand(42);  // Fixed seed for deterministic centroid initialization

    std::vector<double> hx, hy, hz;

    if (!load_data("spotify_pca_cleaned.csv", hx, hy, hz)) {
        printf("\u274c Failed to load data\n");
        return 1;
    }

    int n = hx.size();
    printf("\u2705 Loaded %d data points\n", n);

    std::vector<int> h_labels(n, 0);

    // Allocate device memory for coordinates and centroid computation
    double *dx, *dy, *dz, *dcx, *dcy, *dcz, *dsx, *dsy, *dsz;
    int *d_labels, *d_count;

    cudaMalloc(&dx, n * sizeof(double));
    cudaMalloc(&dy, n * sizeof(double));
    cudaMalloc(&dz, n * sizeof(double));
    cudaMalloc(&dcx, N_CLUSTERS * sizeof(double));
    cudaMalloc(&dcy, N_CLUSTERS * sizeof(double));
    cudaMalloc(&dcz, N_CLUSTERS * sizeof(double));
    cudaMalloc(&dsx, N_CLUSTERS * sizeof(double));
    cudaMalloc(&dsy, N_CLUSTERS * sizeof(double));
    cudaMalloc(&dsz, N_CLUSTERS * sizeof(double));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_count, N_CLUSTERS * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(dx, hx.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    // Try different block sizes to evaluate performance (scaling study)
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        // Deterministic initial centroid selection (even spacing)
        std::vector<double> cx(N_CLUSTERS), cy(N_CLUSTERS), cz(N_CLUSTERS);
        for (int k = 0; k < N_CLUSTERS; ++k) {
            int idx = k * (n / N_CLUSTERS);
            cx[k] = hx[idx];
            cy[k] = hy[idx];
            cz[k] = hz[idx];
        }

        cudaMemcpy(dcx, cx.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dcy, cy.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dcz, cz.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);

        int threads = block_size;
        int blocks = (n + threads - 1) / threads;

        auto start = std::chrono::high_resolution_clock::now();

        // Perform k-means iterations
        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            assign_clusters<<<blocks, threads>>>(dx, dy, dz, dcx, dcy, dcz, d_labels, n);
            reset_centroids<<<1, N_CLUSTERS>>>(dsx, dsy, dsz, d_count);
            accumulate_centroids<<<blocks, threads>>>(dx, dy, dz, d_labels, dsx, dsy, dsz, d_count, n);

            cudaMemcpy(cx.data(), dsx, N_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cy.data(), dsy, N_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cz.data(), dsz, N_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
            std::vector<int> count(N_CLUSTERS);
            cudaMemcpy(count.data(), d_count, N_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);

            // Update centroids on host
            for (int k = 0; k < N_CLUSTERS; ++k) {
                if (count[k] > 0) {
                    cx[k] /= count[k];
                    cy[k] /= count[k];
                    cz[k] /= count[k];
                }
            }

            cudaMemcpy(dcx, cx.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dcy, cy.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dcz, cz.data(), N_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
        }

        // Copy final cluster assignments back to host
        cudaMemcpy(h_labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        printf("\u23F1 Block size %d: %.4f seconds\n", block_size, duration.count());

        // Save clustering result with cluster IDs
        std::ostringstream filename;
        filename << "clustered_output_block" << block_size << ".csv";
        std::ofstream out(filename.str());
        out << "id,x,y,z,cluster\n";
        for (int i = 0; i < n; ++i) {
            out << i << "," << std::fixed << std::setprecision(6)
                << hx[i] << "," << hy[i] << "," << hz[i] << "," << h_labels[i] << "\n";
        }
        printf("\u2705 Saved output to %s\n", filename.str().c_str());
    }

    // Free device memory
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    cudaFree(dcx); cudaFree(dcy); cudaFree(dcz);
    cudaFree(dsx); cudaFree(dsy); cudaFree(dsz);
    cudaFree(d_labels); cudaFree(d_count);

    return 0;
}

