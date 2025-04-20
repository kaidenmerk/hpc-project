# 1. Build & Execution
First, download the `spotify_pca_cleaned.csv` dataset in `spotify_data/data_links.txt`
## Serial Build & Execution
```
module load gcc
cd serial\
make
./kmeans_serial ../spotify_pca_cleaned.csv 9 100 ../outputs/serial_output.csv
```
Example usage shown above to replicate our approach with K = 9 and Epochs = 100.

## Shared Memory CPU Build & Execution

```
module load gcc
cd shared\
make
./kmeans_cpu_shared ../spotify_pca_cleaned.csv 9 100 4 ../outputs/cpu_shared_output.csv
```

Example usage shown above to replicate our approach with K = 9, Epochs = 100 and num_threads = 4.

# 2. Approaches
Data Cleaning: Due to the high dimensionality of our data, we decided to use the PCA algorithm to compress numeric fields we found relevant to genre (Insert used features here). We compressed the N dimensional data into 3 dimensions (x, y, z) and used that for our clustering.

1. Serial: For the serial approach, we used Lloyd's algorithm as described in the given tutorial. For each point, compute the Euclidean squared distance to all centroids or clusters, and assign it to the nearest one. Then recompute each centroid to be the mean of its included points. This is done over n points across m epochs.
2. Parallel Shared Memory CPU: We optimized the serial approach using openMP. To do this, we split the dataset accross the T threads we had available, so each thread was in charge of a subset of points. Each thread maintained their own arrays for each clusters. After a barrier, the individual arrays were combined, and the clusters were updated.
3. Parallel CUDA GPU
4. Distributed Memory CPU
5. Distributed Memory GPU

# 3. Scaling Study Results

## Shared Memory CPU

**Setup:**
- K = 9, max_iters = 100
- Timing via shell 'time' of full end-to-end run

**Results:**
| Threads | Wall‑Clock (s) | Speedup |
|:-------:|:--------------:|:-------:|
| 1       | 4.025          | ----    |
| 2       | 2.997          | 1.34    |
| 3       | 2.550          | 1.58    |
| 4       | 2.498          | 1.61    |
| 5       | 2.486          | 1.62    |
# 4. Validation
After running 2 or more implementations, validation can be performed using
```python validate.py path-to-directory```
where the .csv files to be checked are stored in the same directory.

# 5. Visualization
  ![30°/45°](visualizations/clusters_e30_a45.png)

  ![90°/0°](visualizations/clusters_e90_a0.png)

  ![0°/0°](visualizations/clusters_e0_a0.png)

  ![0°/90°](visualizations/clusters_e0_a90.png)
# 6. Who did what

Kaiden McMillen did the serial and parallel shared memory CPU implementations and their scaling studies. Kaiden also handled the validation and visualization of the implementations.
