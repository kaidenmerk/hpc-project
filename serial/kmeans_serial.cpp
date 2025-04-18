// serial/kmeans_serial.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <stdexcept>

using FeatureVec = std::vector<float>;

struct Point {
    FeatureVec x;
    int cluster{-1};
    Point(size_t D = 0) : x(D) {}
};

// Load CSV assuming all columns are numeric
std::vector<Point> load_data(const std::string &path, size_t &D) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Error opening file: " << path << "\n";
        std::exit(1);
    }
    std::string line;
    // 1) Read header to count columns
    if (!std::getline(fin, line)) {
        std::cerr << "Empty file: " << path << "\n";
        std::exit(1);
    }
    std::stringstream hdr(line);
    std::string field;
    D = 0;
    while (std::getline(hdr, field, ',')) {
        D++;
    }
    if (D == 0) {
        std::cerr << "No columns found in header of " << path << "\n";
        std::exit(1);
    }

    // 2) Read data rows
    std::vector<Point> data;
    data.reserve(1200000);
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        Point p(D);
        for (size_t i = 0; i < D; ++i) {
            if (!std::getline(ss, field, ',')) {
                std::cerr << "Missing field at row " << (data.size()+2)
                          << ", col " << i << "\n";
                std::exit(1);
            }
            try {
                p.x[i] = std::stof(field);
            } catch (const std::invalid_argument &) {
                std::cerr << "Non-numeric value '" << field
                          << "' at row " << (data.size()+2)
                          << ", col " << i << "\n";
                std::exit(1);
            }
        }
        data.push_back(std::move(p));
    }
    return data;
}

// Randomly pick K initial centroids from data
std::vector<FeatureVec> init_centroids(
    const std::vector<Point> &data,
    size_t K, std::mt19937 &rng)
{
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    size_t D = data[0].x.size();
    std::vector<FeatureVec> cent(K, FeatureVec(D));
    for (size_t j = 0; j < K; ++j) {
        cent[j] = data[dist(rng)].x;
    }
    return cent;
}

// Squared Euclidean distance
def float sqrDist(const FeatureVec &a, const FeatureVec &b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Lloyd's algorithm
void kmeans(std::vector<Point> &data,
            std::vector<FeatureVec> &centroids,
            size_t max_iters, float tol)
{
    size_t K = centroids.size();
    size_t D = centroids[0].size();
    std::vector<FeatureVec> newc(K, FeatureVec(D));
    std::vector<size_t> counts(K);

    for (size_t iter = 0; iter < max_iters; ++iter) {
        // zero accumulators
        for (size_t j = 0; j < K; ++j) {
            std::fill(newc[j].begin(), newc[j].end(), 0.0f);
            counts[j] = 0;
        }
        // assign + accumulate
        for (auto &pt : data) {
            size_t best = 0;
            float bestD = sqrDist(pt.x, centroids[0]);
            for (size_t j = 1; j < K; ++j) {
                float d = sqrDist(pt.x, centroids[j]);
                if (d < bestD) { bestD = d; best = j; }
            }
            pt.cluster = best;
            counts[best]++;
            for (size_t d = 0; d < D; ++d)
                newc[best][d] += pt.x[d];
        }
        // update centroids & check movement
        float maxMove = 0.0f;
        for (size_t j = 0; j < K; ++j) {
            for (size_t d = 0; d < D; ++d)
                newc[j][d] /= counts[j];
            maxMove = std::max(maxMove,
                               sqrDist(centroids[j], newc[j]));
            centroids[j] = newc[j];
        }
        if (maxMove < tol) break;
    }
}

// Write out id,cluster pairs
void write_labels(const std::string &out_path,
                  const std::vector<Point> &data)
{
    std::ofstream fout(out_path);
    fout << "id,cluster\n";
    for (size_t i = 0; i < data.size(); ++i)
        fout << i << "," << data[i].cluster << "\n";
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: kmeans_serial <csv> <K> <tol> <max_iters> <out.csv>\n";
        return 1;
    }
    std::string csv = argv[1];
    size_t K        = std::stoul(argv[2]);
    float tol       = std::stof(argv[3]);
    size_t iters    = std::stoul(argv[4]);
    std::string out = argv[5];

    size_t D;
    auto data = load_data(csv, D);

    std::mt19937 rng(42);
    auto centroids = init_centroids(data, K, rng);

    kmeans(data, centroids, iters, tol);
    write_labels(out, data);
    return 0;
}
