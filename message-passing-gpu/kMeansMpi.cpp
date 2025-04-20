#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>

typedef struct {
    double x, y, z, minDist;
    int cluster;
} Point;

extern "C" {void launchDistance(Point *data, Point *centroids, int itemNum);}

int getFileLength() {
    int temp = 0;
    std::string line;
    std::ifstream file("test.csv");
    while(getline(file,line)) {
        temp++;
    }
    return (temp-1);
}

void readCsv(Point data[]) {
    int lineNum = 0;
    bool first = true;
    std::vector<Point> points;
    std::string line;
    std::ifstream file("test.csv");
    while (getline(file, line)) {
        if(first) {
            first = false;
            continue;
        }
        lineNum++;
        std::vector<std::string> t;
        std::string str;
        std::istringstream ss(line);
        while(getline(ss, str, ',')) {
            t.push_back(str);
        }

        Point temp;
        temp.x = std::stod(t[0].data());
        temp.y = std::stod(t[1].data());
        temp.z = std::stod(t[2].data());
        temp.minDist = __DBL_MAX__;
        temp.cluster = -1;
        data[lineNum] = temp;
    }
}



int main(int argc, char **argv)
{
    int epoch = 100;
    int k = 9;
    int itemNum = 0;


    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //sets up our custom mpi type
    const int nItems = 5;
    int blocklengths[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Datatype mpi_point_type;
    MPI_Aint offsets[5];
    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);
    offsets[2] = offsetof(Point, z);
    offsets[3] = offsetof(Point, minDist);
    offsets[4] = offsetof(Point, cluster);


    MPI_Type_create_struct(nItems, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Point *data = NULL;
    Point *centroidData = NULL;

    Point *recvPointData = NULL;

    if(!rank) {

        itemNum = getFileLength();
        //read elements in
        centroidData = (Point *)malloc(sizeof(Point) * k);
        data = (Point *)malloc(sizeof(Point) * itemNum);
        //i don't know how to use vectors with mpi
        readCsv(data);
        
        srand(0);
        for(int i = 0; i < k; i++) {
            int temp = rand() % itemNum;
            centroidData[i].x = data[temp].x;
            centroidData[i].y = data[temp].y;
            centroidData[i].z = data[temp].z;
            centroidData[i].cluster = i;
        }
        MPI_Bcast(&itemNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

        recvPointData = (Point *)malloc(sizeof(Point) * ((itemNum / size) + 1));
    } else {
        MPI_Bcast(&itemNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

        centroidData = (Point *)malloc(sizeof(Point) * k);
        recvPointData = (Point *)malloc(sizeof(Point) * ((itemNum / size) + 1));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < epoch; i++) {
        MPI_Bcast(centroidData, k, mpi_point_type, 0, MPI_COMM_WORLD);

        MPI_Scatter(data, ((itemNum / size) + 1), mpi_point_type, recvPointData, ((itemNum / size) + 1), mpi_point_type, 0, MPI_COMM_WORLD);

        launchDistance(recvPointData, centroidData, ((itemNum / size) + 1));

        MPI_Gather(recvPointData, ((itemNum / size) + 1), mpi_point_type, data, ((itemNum / size) + 1), mpi_point_type, 0, MPI_COMM_WORLD);

        //compute new centroids
        if(!rank) {
            double sumX;
            double sumY;
            double sumZ;

            int total;

            for(int j = 0; j < k; j++) {
                sumX = 0;
                sumY = 0;
                sumZ = 0;

                total = 0;

                for(int l = 0; l < itemNum; l++) {
                    if(data[l].cluster == j) {
                        sumX += data[l].x;
                        sumY += data[l].y;
                        sumZ += data[l].z;
                        total++;
                    }
                }
                if(total > 0) {
                    centroidData[j].x += sumX / total;
                    centroidData[j].y += sumY / total;
                    centroidData[j].z += sumZ / total;
                }
            }
        }
    }

    if(!rank) {
        auto end =std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::vector<Point> points;
        for(int i = 0; i < itemNum; i++) {
            points.push_back(data[i]);
        }
        


        std::cout << "We're done :)" << std::endl;
        std::cout << "It took " << duration.count() << " milliseconds" << std::endl;
        std::ofstream myfile;
        myfile.open("output.csv");
        myfile << "x, y , z" << std::endl;

        for (std::vector<Point>::iterator it = points.begin(); it != points.end();
             ++it) {
            myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
        }
        myfile.close();
    }

    free(data);
    free(centroidData);
    free(recvPointData);

    MPI_Type_free(&mpi_point_type);
    MPI_Finalize();

    return 0;
}
