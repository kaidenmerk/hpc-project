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

1. Serial
2. Parallel Shared Memory CPU
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
After running 2 or more implementations, validation can be performed using \\
\' python validate.py path-to-directory \' \\
where the .csv files to be checked are stored in the same directory.

# 5. Visualization

# 6. Who did what

Kaiden McMillen did the serial and parallel shared memory CPU implementations and their scaling studies. Kaiden also handled the validation and visualization of the implementations.
